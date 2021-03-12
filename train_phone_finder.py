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
        "folder",
        type=str,
        help="Path to a folder containing training images and label.txt",
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
        default=0.3,
        help="Saliency map masking threshold (default: 0.3)",
    )
    parser.add_argument(
        "--min-window-size",
        type=int,
        default=64,
        help="Min window size (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="Discount factor for training (default: 0.999)",
    )
    parser.add_argument(
        "--eps-start",
        type=float,
        default=0.01,
        help="Starting epsilon of epsilon-greedy (default: 0.9)",
    )
    parser.add_argument(
        "--eps-end",
        type=float,
        default=0.001,
        help="Final epsilon of epsilon-greedy (default: 0.05)",
    )
    parser.add_argument(
        "--eps-num",
        type=int,
        default=300,
        help="Number of steps to decay from starting epsilon to final epsilon (default: 300)",
    )
    parser.add_argument(
        "--target-update",
        type=int,
        default=10,
        help="Number of episodes to update the target network (default: 10)",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=1000,
        help="Size of replay memory (default: 1000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate of Adam (default: 1e-6)",
    )
    parser.add_argument(
        "--max-num-episodes",
        type=int,
        default=10000,
        help="Max number of episodes (default: 10000)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Checkpoint saving interval (default: 50 episodes)",
    )
    parser.add_argument(
        "--render",
        type=int,
        default=0,
        help="Visualize the training process? (default: 0)",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        default=1,
        help="Load checkpoint and continue training? (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    args = parser.parse_args()
    return args


def load_images_and_positions(args):
    """Load images and phone positions from the given folder """
    with open(args.folder + "/labels.txt") as f:
        lines = f.read().splitlines()
    images, positions = [], []
    for line in lines:
        file_name, x, y = line.split()
        position = (float(x), float(y))
        image = cv2.imread(args.folder + "/" + file_name)
        images.append(image)
        positions.append(position)
    return images, positions


def euclidean_dist(point_a, point_b):
    return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


class Env(object):
    def __init__(self, images, positions, args):
        self.images = images
        self.positions = positions
        self.ratio = args.ratio
        self.device = args.device
        self.env_index = 0  # Which image are we using?
        self.update_image_and_goal()
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
        self.episode_total_rewards = []
        self.action_space = np.arange(5)

    def update_image_and_goal(self):
        self.env_index = (self.env_index + 1) % len(self.images)
        self.goal = self.positions[self.env_index]
        image = self.images[self.env_index]
        spectral_residual = cv2.saliency.StaticSaliencySpectralResidual_create()
        _, saliency_map = spectral_residual.computeSaliency(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image[saliency_map < 0.3] = 0
        self.image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        self.max_row = self.image.shape[0] - 1
        self.max_col = self.image.shape[1] - 1

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
        self.update_image_and_goal()
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
        center = self.get_center()
        dist = euclidean_dist(center, self.goal)
        if dist < 0.05:  # If agent arrives at the goal position
            reward = 1.0
        else:
            reward = -dist
        if euclidean_dist(self.upper_left, self.lower_right) < 0.05:
            done = True
        return next_state, reward, done

    def render(self, done=False):
        if self.ax is None:
            self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 5))
        self.ax[0].cla()
        self.ax[0].imshow(self.image, aspect="auto")
        self.ax[0].plot(self.goal[0] * self.max_col, self.goal[1] * self.max_row, "r*")
        start_row, end_row, start_col, end_col = self.get_indices()
        self.ax[0].vlines(start_col, ymin=start_row, ymax=end_row, color="r", lw=3)
        self.ax[0].vlines(end_col, ymin=start_row, ymax=end_row, color="r", lw=3)
        self.ax[0].hlines(start_row, xmin=start_col, xmax=end_col, color="r", lw=3)
        self.ax[0].hlines(end_row, xmin=start_col, xmax=end_col, color="r", lw=3)
        center = self.get_center()
        self.ax[0].plot(center[0] * self.max_col, center[1] * self.max_row, "gs")
        if done:
            self.ax[1].cla()
            rewards_t = torch.tensor(self.episode_total_rewards, dtype=torch.float)
            self.ax[1].set_title("Training...")
            self.ax[1].set_xlabel("Episode")
            self.ax[1].set_ylabel("Total rewards")
            self.ax[1].plot(rewards_t.numpy())
            # Take 100 episode averages and plot them too
            if len(rewards_t) >= 100:
                means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                self.ax[1].plot(means.numpy())
        plt.pause(1 / 60)


def epsilon_greedy(state, policy_net, steps_done, args):
    sample = random.random()
    epsilon = args.eps_end + (args.eps_start - args.eps_end) * np.exp(
        -steps_done / args.eps_num
    )
    steps_done += 1
    if sample > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), steps_done
    else:
        return (
            torch.tensor([[random.randrange(5)]], device=args.device, dtype=torch.long),
            steps_done,
        )


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

    # Collect trainable parameters
    trainable_params = []
    for param in policy_net.parameters():
        if param.requires_grad:
            trainable_params.append(param)
    return policy_net.to(args.device), trainable_params


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


def optimize_policy(memory, policy_net, target_net, optimizer, args):
    if len(memory) < args.batch_size:
        return
    transitions = memory.sample(args.batch_size)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=args.device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(args.batch_size, device=args.device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch
    policy_net.train()
    loss = torch.nn.functional.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )
    optimizer.zero_grad()
    loss.backward(retain_graph=False)
    optimizer.step()
    policy_net.eval()


def main():
    args = parse_arguments()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device

    # Create environment
    images, positions = load_images_and_positions(args)
    env = Env(images, positions, args)

    # Build model
    policy_net, trainable_params = build_policy_net(args)

    # For training
    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)
    memory = ReplayMemory(args.memory_size)
    steps_done = 0
    if args.load_checkpoint:
        checkpoint = torch.load("./checkpoint.pt")
        steps_done = checkpoint["steps_done"]
        env.episode_total_rewards = checkpoint["episode_total_rewards"]
        policy_net.load_state_dict(checkpoint["policy_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        memory = checkpoint["memory"]
    target_net = copy.deepcopy(policy_net)
    target_net.eval()

    for episode_index in range(args.max_num_episodes):
        state = env.reset()
        if args.render:
            env.render()
        total_reward = 0.0
        while True:
            action, steps_done = epsilon_greedy(state, policy_net, steps_done, args)
            next_state, reward, done = env.step(action)
            total_reward += reward
            reward = torch.tensor([reward], device=args.device, dtype=torch.float32)
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_policy(memory, policy_net, target_net, optimizer, args)
            if args.render:
                env.render(done)
            if done:
                env.episode_total_rewards.append(total_reward)
                print(f"Episode {episode_index:d} total reward {total_reward:.2f}")
                if episode_index % args.save_interval or total_reward > np.max(
                    env.episode_total_rewards
                ):
                    torch.save(
                        {
                            "steps_done": steps_done,
                            "episode_total_rewards": env.episode_total_rewards,
                            "policy_state_dict": policy_net.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "memory": memory,
                        },
                        "checkpoint.pt",
                    )
                break


if __name__ == "__main__":
    main()
