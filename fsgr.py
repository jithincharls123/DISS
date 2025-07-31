"""
Few-shot dynamic gesture recognition using graph and temporal convolutional
networks with a relation network classifier.

This script reimplements the personalised few-shot gesture recogniser using
a spatiotemporal backbone inspired by recent research. Graph-based models
can capture spatial relationships between hand keypoints by computing
joint-to-joint and joint-to-group attentions【310899056794082†L224-L229】.
Temporal dependencies are encoded by dilated convolutional networks and
attention mechanisms【877238131962073†L118-L122】.  Our model replaces the
prototype-based classifier with a relation network that learns to score
the similarity between query and support embeddings.  Sequences are
resampled to a length of 256 frames to provide finer temporal
resolution.  Hyper-parameters such as the GCN output dimension, TCN
channel sizes and relation-network hidden size are configurable via
command-line options.

Key features:

* **Graph convolution:** Each frame’s 21×3 landmark tensor is processed by a
  Graph Convolutional Network (GCN).  This explicitly models the
  adjacency between joints.
* **Temporal convolution:** A stack of dilated 1D convolutions captures
  short- and long-term temporal dependencies, akin to temporal
  convolutional networks.
* **Relation network classifier:** Rather than averaging support embeddings
  into prototypes, a small neural network learns to score query-support
  embeddings.  Class scores are obtained by averaging relation scores
  across support examples per class.
* **Resampling length of 256:** Sequences are resampled to 256 frames by
  default, providing finer temporal resolution.

The script preserves the asynchronous video capture, landmark extraction,
windowing, and smoothing from the previous version.  Use the command
line arguments to tune hyper-parameters such as the GCN output
dimension, temporal convolution channels, and embedding size.
"""

import argparse
import logging
import threading
from collections import Counter, deque
from typing import List, Tuple

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration constants
CONFIG = {
    'pos_noise_std': 0.015,
    'smooth_k': 7,
    'input_dim': 21 * 3,
    'mp_det_conf': 0.5,
    'mp_track_conf': 0.5,
    'window_timeout': 1.0,
    'ema_alpha': 0.3,
    'support_timeout': 30.0,
}


def resample_sequence(seq: torch.Tensor, target: int) -> torch.Tensor:
    """Resample a [T,D] sequence to [target,D] via linear interpolation."""
    T, D = seq.size()
    if T == target:
        return seq
    idx = torch.linspace(0, T - 1, steps=target, device=seq.device)
    i0 = idx.floor().long()
    i1 = torch.clamp(i0 + 1, max=T - 1)
    w = (idx - i0.float()).unsqueeze(1)
    return seq[i0] * (1 - w) + seq[i1] * w


class VideoCaptureAsync:
    """Asynchronous camera capture with buffering."""

    def __init__(self, src: int = 0, width: int | None = None, height: int | None = None,
                 buffer_size: int = 4) -> None:
        self.cap = cv2.VideoCapture(src)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.buffer = deque(maxlen=buffer_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("camera frame drop")
                self.stopped = True
                break
            self.buffer.append(frame)

    def read(self, timeout: float = 1.0) -> Tuple[bool, torch.Tensor | None]:
        if not self.buffer:
            threading.Event().wait(timeout)
        if self.buffer:
            return True, self.buffer.popleft()
        return False, None

    def release(self) -> None:
        self.stopped = True
        self.thread.join()
        self.cap.release()


class LandmarkSmoother:
    """Moving average smoother for landmark coordinates."""

    def __init__(self, window_size: int = 5) -> None:
        self.win = deque(maxlen=window_size)

    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        self.win.append(pts)
        stacked = torch.stack(list(self.win), dim=0)
        return stacked.mean(dim=0)


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
HAND_PROC = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=CONFIG['mp_det_conf'],
    min_tracking_confidence=CONFIG['mp_track_conf'],
)


def extract_landmarks(frame, smoother: LandmarkSmoother) -> torch.Tensor | None:
    """Extract and normalise MediaPipe hand landmarks from a frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = HAND_PROC.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0]
    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
    pts = [c for p in lm.landmark for c in (p.x, p.y, p.z)]
    coords = torch.tensor(pts, dtype=torch.float32)
    wrist = coords[:3]
    coords = coords - wrist.repeat(21)
    maxv = coords.abs().max()
    if maxv > 0:
        coords /= maxv
    return smoother(coords)


def build_adjacency(num_nodes: int = 21) -> torch.Tensor:
    """Construct a normalised adjacency matrix for the MediaPipe hand skeleton."""
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # index
        (5, 9), (9, 10), (10, 11), (11, 12), # middle
        (9, 13), (13, 14), (14, 15), (15, 16), # ring
        (13, 17), (17, 18), (18, 19), (19, 20), # pinky
        (0, 17), (0, 13), (0, 9), (0, 5),    # palm connections
    ]
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i, j in connections:
        A[i, j] = 1
        A[j, i] = 1
    A += torch.eye(num_nodes, dtype=torch.float32)
    deg = A.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


class GraphConv(nn.Module):
    """Graph convolution layer using a fixed adjacency matrix."""

    def __init__(self, in_dim: int, out_dim: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.adj = adjacency  # [N,N]
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, F]
        B, T, N, F = x.shape
        A = self.adj.to(x.device)  # [N,N]
        x = x.reshape(B * T, N, F)
        out = A @ x @ self.weight  # [B*T, N, out_dim]
        out = out + self.bias
        return out.reshape(B, T, N, -1)


class TemporalBlock(nn.Module):
    """Single layer of a temporal convolutional network with dilation and residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        self.pad_cut = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        out = self.conv(x)
        out = out[:, :, :-self.pad_cut] if self.pad_cut > 0 else out
        out = self.relu(out)
        res = x if self.downsample is None else self.downsample(x)
        if res.shape[-1] > out.shape[-1]:
            res = res[:, :, :out.shape[-1]]
        elif res.shape[-1] < out.shape[-1]:
            pad = out.shape[-1] - res.shape[-1]
            res = F.pad(res, (pad, 0))
        return out + res


class TemporalConvNet(nn.Module):
    """Stack of temporal blocks for sequence modelling."""

    def __init__(self, num_inputs: int, channels: List[int], kernel_size: int = 3) -> None:
        super().__init__()
        layers = []
        dilation = 1
        in_ch = num_inputs
        for out_ch in channels:
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation))
            in_ch = out_ch
            dilation *= 2
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        return self.network(x)


class STEncoder(nn.Module):
    """Spatiotemporal encoder combining graph and temporal convolutions."""

    def __init__(self, gcn_in_dim: int, gcn_out_dim: int, tcn_channels: List[int], emb_dim: int,
                 adjacency: torch.Tensor) -> None:
        super().__init__()
        self.gcn = GraphConv(gcn_in_dim, gcn_out_dim, adjacency)
        tcn_input_dim = gcn_out_dim * adjacency.size(0)
        self.tcn = TemporalConvNet(tcn_input_dim, tcn_channels)
        self.fc = nn.Linear(tcn_channels[-1], emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N*F] where F=gcn_in_dim
        B, T, _ = x.shape
        N = int(x.shape[2] / self.gcn.weight.shape[0])
        x = x.view(B, T, N, -1)
        x = self.gcn(x)  # [B, T, N, gcn_out_dim]
        x = x.view(B, T, -1)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.mean(dim=2)
        emb = F.normalize(self.fc(x), dim=1)
        return emb


class RelationNetwork(nn.Module):
    """Relation network for computing similarity between embeddings."""

    def __init__(self, emb_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        B, E = query.shape
        S = support.size(0)
        q_exp = query.unsqueeze(1).expand(B, S, E)
        s_exp = support.unsqueeze(0).expand(B, S, E)
        x = torch.cat([q_exp, s_exp], dim=2)
        x = self.relu(self.fc1(x))
        scores = self.fc2(x).squeeze(-1)
        return scores


class DynamicFewShotRecognizer:
    """Encapsulates support capture and live inference using GCN-TCN and relation network."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cap = VideoCaptureAsync(src=args.camera)
        self.smoother = LandmarkSmoother(window_size=5)
        self.adj = build_adjacency()
        self.encoder = STEncoder(
            gcn_in_dim=3,
            gcn_out_dim=args.gcn_out_dim,
            tcn_channels=args.tcn_channels,
            emb_dim=args.emb_dim,
            adjacency=self.adj,
        ).to(self.device)
        self.encoder.eval()
        if self.device.type == 'cpu':
            self.encoder = torch.quantization.quantize_dynamic(
                self.encoder, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
            )
        self.relnet = RelationNetwork(args.emb_dim, hidden_dim=args.relnet_hidden)
        self.relnet.to(self.device)
        self.relnet.eval()
        self.support_embeds: List[torch.Tensor] = []
        self.support_labels: List[int] = []
        self.names: List[str] = []

    @property
    def device(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def capture_support(self, names: List[str], shots: int, seq_len: int,
                        augment_times: int) -> bool:
        self.names = names
        embeds: List[torch.Tensor] = []
        labels: List[int] = []
        cv2.namedWindow('Support Capture', cv2.WINDOW_NORMAL)
        try:
            for idx, name in enumerate(names):
                logger.info("press 's' to start support capture for %s", name)
                while True:
                    ret, frm = self.cap.read()
                    if not ret:
                        continue
                    cv2.putText(frm, f"Ready for '{name}', hit 's'",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow('Support Capture', frm)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        break
                    if key == ord('q'):
                        self.cap.release()
                        cv2.destroyAllWindows()
                        return False
                for shot in range(1, shots + 1):
                    buf: List[torch.Tensor] = []
                    t0 = cv2.getTickCount()
                    while len(buf) < seq_len:
                        ret, frm = self.cap.read()
                        if not ret:
                            continue
                        lm = extract_landmarks(frm, self.smoother)
                        if lm is not None:
                            buf.append(lm)
                        cv2.putText(frm, f"Capturing '{name}' {len(buf)}/{seq_len}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.imshow('Support Capture', frm)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.cap.release()
                            cv2.destroyAllWindows()
                            return False
                        if ((cv2.getTickCount() - t0) / cv2.getTickFrequency() > CONFIG['support_timeout']):
                            logger.warning("timed out on %s shot %d", name, shot)
                            break
                    if buf:
                        seq = resample_sequence(torch.stack(buf), seq_len)
                        seq = seq.view(seq.size(0), -1)
                        with torch.inference_mode():
                            emb = self.encoder(seq.unsqueeze(0).to(self.device))
                        embeds.append(emb.squeeze(0))
                        labels.append(idx)
                        for _ in range(augment_times):
                            noisy = seq + torch.randn_like(seq) * CONFIG['pos_noise_std']
                            with torch.inference_mode():
                                emb_aug = self.encoder(noisy.unsqueeze(0).to(self.device))
                            embeds.append(emb_aug.squeeze(0))
                            labels.append(idx)
        finally:
            cv2.destroyWindow('Support Capture')
        if not embeds:
            logger.error("no support data captured")
            return False
        self.support_embeds = embeds
        self.support_labels = labels
        self._train_relation_network()
        return True

    def _train_relation_network(self, epochs: int = 50, lr: float = 0.01) -> None:
        """Fine-tune the relation network on pairs of support embeddings."""
        if len(self.support_embeds) < 2:
            return
        device = self.device
        embeds = torch.stack(self.support_embeds).to(device)
        labels = torch.tensor(self.support_labels, device=device)
        pair_x = []
        pair_y = []
        for i in range(len(embeds)):
            for j in range(len(embeds)):
                if i == j:
                    continue
                pair_x.append(torch.cat([embeds[i], embeds[j]], dim=0))
                pair_y.append(1.0 if labels[i] == labels[j] else 0.0)
        x = torch.stack(pair_x).to(device)
        y = torch.tensor(pair_y, device=device).unsqueeze(1)
        criterion = nn.BCEWithLogitsLoss()
        optimiser = torch.optim.SGD(self.relnet.parameters(), lr=lr)
        self.relnet.train()
        for _ in range(epochs):
            optimiser.zero_grad()
            z = self.relnet.relu(self.relnet.fc1(x))
            logits = self.relnet.fc2(z)
            loss = criterion(logits, y)
            loss.backward()
            optimiser.step()
        self.relnet.eval()

    def run_live(self, seq_len: int, min_len: int, conf_thresh: float) -> None:
        if not self.support_embeds:
            logger.error("run capture_support() first")
            return
        support_tensor = torch.stack(self.support_embeds).to(self.device)
        support_labels = torch.tensor(self.support_labels, dtype=torch.long, device=self.device)
        window: deque[torch.Tensor] = deque()
        last_detect_time = cv2.getTickCount()
        pred_buf: deque[int] = deque(maxlen=CONFIG['smooth_k'])
        ema_prob: float = 0.0
        cv2.namedWindow('Live', cv2.WINDOW_NORMAL)
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                lm = extract_landmarks(frame, self.smoother)
                now = (cv2.getTickCount() - last_detect_time) / cv2.getTickFrequency()
                if lm is not None:
                    window.append(lm)
                    last_detect_time = cv2.getTickCount()
                    if len(window) > seq_len:
                        window.popleft()
                if window and now > CONFIG['window_timeout']:
                    window.clear()
                    pred_buf.clear()
                    ema_prob = 0.0
                if len(window) >= min_len:
                    seq = resample_sequence(torch.stack(list(window)), seq_len)
                    seq_flat = seq.view(seq.size(0), -1).unsqueeze(0).to(self.device)
                    with torch.inference_mode():
                        q_emb = self.encoder(seq_flat)
                    rel_scores = self.relnet(q_emb, support_tensor)
                    class_scores = []
                    for i in range(len(self.names)):
                        mask = (support_labels == i)
                        if mask.any():
                            class_scores.append(rel_scores[:, mask].mean(dim=1))
                        else:
                            class_scores.append(torch.tensor([float('-inf')], device=self.device))
                    class_scores = torch.stack(class_scores, dim=1)
                    probs = F.softmax(class_scores, dim=1).cpu().squeeze(0)
                    conf_tensor, idx_tensor = probs.max(dim=0)
                    conf = conf_tensor.item()
                    idx = idx_tensor.item()
                    ema_prob = ema_prob * CONFIG['ema_alpha'] + conf * (1 - CONFIG['ema_alpha'])
                    raw_label = self.names[idx] if conf > conf_thresh else 'Unknown'
                    pred_buf.append(idx)
                    if len(pred_buf) == CONFIG['smooth_k']:
                        most_common_idx = Counter(pred_buf).most_common(1)[0][0]
                        label = self.names[most_common_idx] if conf > conf_thresh else 'Unknown'
                    else:
                        label = raw_label
                    cv2.putText(frame, f"{label}: {ema_prob * 100:.1f}%",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Collecting frames: {len(window)}/{min_len}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow('Live', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description='Graph-TCN few-shot gesture recogniser')
    parser.add_argument('--camera', type=int, default=0, help='Index of camera to use')
    parser.add_argument('--shots', type=int, default=6, help='Number of support samples per class')
    parser.add_argument('--seq_len', type=int, default=256, help='Resampled sequence length')
    parser.add_argument('--min_len', type=int, default=None, help='Minimum window size before inference')
    parser.add_argument('--gcn_out_dim', type=int, default=32, help='Output dimension of GCN layer')
    parser.add_argument('--tcn_channels', type=int, nargs='+', default=[128, 256, 256],
                        help='List of channels for TCN layers')
    parser.add_argument('--emb_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--relnet_hidden', type=int, default=128, help='Hidden dimension of relation network')
    parser.add_argument('--augment', type=int, default=3, help='Number of noise augmentations per support sample')
    parser.add_argument('--conf_thresh', type=float, default=0.45, help='Confidence threshold for predictions')
    args, _ = parser.parse_known_args()
    args.min_len = args.min_len or (args.seq_len // 2)
    recogniser = DynamicFewShotRecognizer(args)
    names = [n.strip() for n in input("Gestures (comma separated): ").split(',') if n.strip()]
    ok = recogniser.capture_support(names, args.shots, args.seq_len, args.augment)
    if not ok:
        logger.error("support capture aborted")
        return
    recogniser.run_live(args.seq_len, args.min_len, args.conf_thresh)


if __name__ == '__main__':
    main()