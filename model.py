def make_model():
    r = 3
    model = nn.Sequential(
        nn.Conv2d(1, 16, r, padding=r//2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, r, padding=r//2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, r, padding=r//2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        layers.LSTM2(64, 32),
        nn.Conv2d(64, 32, 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        layers.LSTM2(32, 32),
        nn.Conv2d(64, 1, 1),
        nn.Sigmoid()
    )
    return model
