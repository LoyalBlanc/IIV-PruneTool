import torch
import torchvision
import torchvision.transforms as transforms


class DataPreFetcher(object):
    def __init__(self, batch_size, train=True):
        self.stream = torch.cuda.Stream()
        train_dataset = torchvision.datasets.MNIST(root='data',
                                                   train=train, transform=transforms.ToTensor(), download=True)
        self.loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=train)
        self.next_image = None
        self.next_label = None
        self.fetcher = None

    def refresh(self):
        self.fetcher = iter(self.loader)
        self.preload()

    def preload(self):
        try:
            self.next_image, self.next_label = next(self.fetcher)
        except StopIteration:
            self.next_image = None
            self.next_label = None
            return
        with torch.cuda.stream(self.stream):
            self.next_image = self.next_image.cuda(device=0, non_blocking=True).float()
            self.next_label = self.next_label.cuda(device=0, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        current_input = self.next_image
        current_target = self.next_label
        if current_input is not None:
            current_input.record_stream(torch.cuda.current_stream())
        if current_target is not None:
            current_target.record_stream(torch.cuda.current_stream())
        self.preload()
        return current_input, current_target


if __name__ == "__main__":
    fetcher = DataPreFetcher(100, train=True)
