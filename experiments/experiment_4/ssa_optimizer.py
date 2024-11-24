import torch


class SSAOptimizer:
    def __init__(self, model, pop_size=30, a=0.5, ST=0.8):
        self.model = model
        self.pop_size = pop_size
        self.a = a
        self.ST = ST
        self.best_position = None
        self.best_fitness = float('inf')
        self.worst_fitness = float('-inf')
        self.current_position = None

        # Initialize population
        self.positions = []
        for _ in range(pop_size):
            position = {}
            for name, param in model.named_parameters():
                position[name] = param.data.clone()
            self.positions.append(position)

        # Store initial position as current best
        self.current_position = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

    def update_detector(self, position, iter_num, itermax):
        R2 = torch.rand(1).item()
        new_position = {}

        for name, param in position.items():
            if R2 < self.ST:
                new_position[name] = param * torch.exp(-torch.tensor(iter_num / (self.a * itermax)))
            else:
                Q = torch.randn_like(param)
                L = torch.ones_like(param)
                new_position[name] = param + Q * L

        return new_position

    def update_follower(self, position, i, best_position, worst_position):
        new_position = {}

        for name, param in position.items():
            if i > self.pop_size / 2:
                Q = torch.randn_like(param)
                new_position[name] = Q * torch.exp((worst_position[name] - param) / (i ** 2))
            else:
                A = torch.randint(0, 2, param.shape) * 2 - 1
                new_position[name] = best_position[name] + torch.abs(param - best_position[name]) * A

        return new_position

    def step(self, criterion, inputs, targets, iter_num, itermax):
        # Initialize worst position and fitness
        worst_position = None
        self.worst_fitness = float('-inf')

        # Evaluate current positions
        for i, position in enumerate(self.positions):
            # Apply position to model
            for name, param in self.model.named_parameters():
                param.data.copy_(position[name])

            # Calculate fitness (loss)
            outputs = self.model(inputs)
            fitness = criterion(outputs, targets.unsqueeze(1)).item()

            # Update best and worst positions
            if self.best_position is None or fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_position = {name: param.clone() for name, param in position.items()}

            if fitness > self.worst_fitness:
                self.worst_fitness = fitness
                worst_position = {name: param.clone() for name, param in position.items()}

        # Ensure we have valid positions before updating
        if self.best_position is None or worst_position is None:
            return

        # Update positions using SSA
        new_positions = []
        n_monitors = int(0.2 * self.pop_size)

        for i in range(self.pop_size):
            if i < n_monitors:
                new_position = self.update_detector(self.positions[i], iter_num, itermax)
            else:
                new_position = self.update_follower(self.positions[i], i, self.best_position, worst_position)
            new_positions.append(new_position)

        self.positions = new_positions

        # Apply best position to model
        for name, param in self.model.named_parameters():
            param.data.copy_(self.best_position[name])

    def zero_grad(self):
        """Compatible with standard optimizer interface"""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def __str__(self):
        return f"SSAOptimizer(pop_size={self.pop_size}, a={self.a}, ST={self.ST})"

