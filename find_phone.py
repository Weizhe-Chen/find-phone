from __future__ import print_function
import argparse
import copy
import random
import torch
from torchvision import transforms, models
from collections import namedtuple
import numpy as np
import cv2
from matplotlib import pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Phone finder training")
    parser.add_argument(
        "path",
        type=str,
        help="Path to a test image",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.25,
        help="Window size reduction ratio (default: 0.25)",
    )
    parser.add_argument(
        "--height", type=int, default=224, help="Resized image height (default: 224)"
    )
    parser.add_argument(
        "--width", type=int, default=224, help="Resized image width (default: 224)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Saliency map masking threshold (default: 0.5)",
    )
    parser.add_argument(
        "--min-window-size",
        type=int,
        default=64,
        help="Min window size (default: 64)",
    )
    parser.add_argument(
        "--render",
        type=int,
        default=0,
        help="Visualize the training process? (default: 0)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    args = parser.parse_args()
    return args


def euclidean_dist(point_a, point_b):
    return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


class Env(object):
    def __init__(self, image, args):
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliency_map) = saliency.computeSaliency(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image[saliency_map < 0.5] = 0
        self.image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        self.max_row = self.image.shape[0] - 1
        self.max_col = self.image.shape[1] - 1
        self.ratio = args.ratio
        self.device = args.device
        self.threshold = args.threshold
        self.upper_left = [0.0, 0.0]
        self.lower_right = [1.0, 1.0]
        self.transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((args.height, args.width)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.356, 0.306], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        num_row_is_larger = self.max_row > self.max_col
        if num_row_is_larger:
            self.min_row_size = args.min_window_size
            self.min_col_size = int(args.min_window_size * self.max_col / self.max_row)
        else:
            self.min_col_size = args.min_window_size
            self.min_row_size = int(args.min_window_size * self.max_row / self.max_col)
        self.fig = None
        self.ax = None
        self.action_space = np.arange(5)

    def get_center(self):
        return [
            (self.upper_left[0] + self.lower_right[0]) / 2,
            (self.upper_left[1] + self.lower_right[1]) / 2,
        ]

    def get_indices(self):
        start_row = int(self.upper_left[1] * self.max_row)
        end_row = int(self.lower_right[1] * self.max_row)
        start_col = int(self.upper_left[0] * self.max_col)
        end_col = int(self.lower_right[0] * self.max_col)
        if (
            end_row - start_row < self.min_row_size
            or end_col - start_col < self.min_col_size
        ):
            center = self.get_center()
            center_row = int(center[1] * self.max_row)
            center_col = int(center[0] * self.max_col)
            start_row = max(0, center_row - self.min_row_size // 2)
            end_row = min(self.max_row, center_row + self.min_row_size // 2)
            start_col = max(0, center_col - self.min_col_size // 2)
            end_col = min(self.max_col, center_col + self.min_col_size // 2)
        return start_row, end_row, start_col, end_col

    def get_current_state(self):
        start_row, end_row, start_col, end_col = self.get_indices()
        window = self.image[start_row:end_row, start_col:end_col, :].transpose(
            (2, 0, 1)
        )
        window = np.ascontiguousarray(window, dtype=np.float32) / 255
        window = torch.from_numpy(window)
        return self.transformer(window).unsqueeze(0).to(self.device)

    def reset(self):
        self.upper_left = [0.0, 0.0]
        self.lower_right = [1.0, 1.0]
        return self.get_current_state()

    def step(self, action):
        done = False
        x_diff = (self.lower_right[0] - self.upper_left[0]) * self.ratio
        y_diff = (self.lower_right[1] - self.upper_left[1]) * self.ratio
        if action == 0:  # Upper left
            self.lower_right[0] -= x_diff
            self.lower_right[1] -= y_diff
        elif action == 1:  # Upper right
            self.upper_left[0] += x_diff
            self.lower_right[1] -= y_diff
        elif action == 2:  # Lower left
            self.lower_right[0] -= x_diff
            self.upper_left[1] += y_diff
        elif action == 3:  # Lower right
            self.upper_left[0] += x_diff
            self.upper_left[1] += y_diff
        elif action == 4:  # Center
            self.upper_left[0] += x_diff / 2
            self.lower_right[0] -= x_diff / 2
            self.upper_left[1] += y_diff / 2
            self.lower_right[1] -= y_diff / 2
        else:
            raise ValueError("Action range: 0, 1, ..., 4")
        next_state = self.get_current_state()
        if euclidean_dist(self.upper_left, self.lower_right) < 0.05:
            done = True
        return next_state, done

    def render(self, done=False):
        if self.ax is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        self.ax.cla()
        self.ax.imshow(self.image, aspect="auto")
        start_row, end_row, start_col, end_col = self.get_indices()
        self.ax.vlines(start_col, ymin=start_row, ymax=end_row, color="r", lw=3)
        self.ax.vlines(end_col, ymin=start_row, ymax=end_row, color="r", lw=3)
        self.ax.hlines(start_row, xmin=start_col, xmax=end_col, color="r", lw=3)
        self.ax.hlines(end_row, xmin=start_col, xmax=end_col, color="r", lw=3)
        center = self.get_center()
        self.ax.plot(center[0] * self.max_col, center[1] * self.max_row, "gs")
        plt.pause(1 / 60)


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def build_policy_net(args):
    # Load pretrained VGG network
    policy_net = models.vgg11_bn(pretrained=True)
    for param in policy_net.parameters():
        param.requires_grad = False

    # Delete average pooling and reinitialize fully connected layers
    # Note that new weights are trainable
    policy_net.avgpool = Identity()
    policy_net.classifier[0] = torch.nn.Linear(
        policy_net.classifier[0].in_features, 512
    )
    policy_net.classifier[2] = Identity()
    policy_net.classifier[3] = torch.nn.Linear(512, 256)
    policy_net.classifier[5] = Identity()
    policy_net.classifier[6] = torch.nn.Linear(256, 5)
    return policy_net.to(args.device)


def main():
    args = parse_arguments()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #  torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device

    # Create environment
    image = cv2.imread(args.path)
    env = Env(image, args)

    # Build model
    policy_net = build_policy_net(args)

    checkpoint = torch.load("./checkpoint.pt")
    policy_net.load_state_dict(checkpoint["policy_state_dict"])

    state = env.reset()
    if args.render:
        env.render()
    while True:
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)
        state, done = env.step(action)
        if args.render:
            env.render(done)
        if done:
            x = (env.upper_left[0] + env.lower_right[0]) / 2
            y = (env.upper_left[1] + env.lower_right[1]) / 2
            print(f"{x:.4f} {y:.4f}")
            break
    plt.show()


if __name__ == "__main__":
    main()
