class BinaryHeap:
    def __init__(self, init_arr: list[int] = None):
        self.heap = []
        self.heap_size = 0
        if init_arr is not None:
            for i in range(len(init_arr)):
                print(self.heap)
                self.add(init_arr[i])

    def __call__(self) -> list[int]:
        return self.heap

    def swap(self, i: int, j: int) -> None:
        dummy = self.heap[i]
        self.heap[i] = self.heap[j]
        self.heap[j] = dummy

    def sift_up(self, i: int) -> None:
        while i >= 1 and self.heap[i] < self.heap[(i - 1) // 2]:
            self.swap(i, (i - 1) // 2)
            i = (i - 1) // 2

    def sift_down(self, i: int) -> None:
        while 2 * i + 1 < self.heap_size:
            left = 2 * i + 1
            right = 2 * i + 2
            j = left
            if right < self.heap_size and self.heap[right] < self.heap[left]:
                j = right

            # check that heap[i] is less or equals both: left and right parents
            if self.heap[i] <= self.heap[j]:
                break

            self.swap(i, j)
            i = j

    def add(self, val: int) -> None:
        self.heap_size += 1
        self.heap.append(val)
        self.sift_up(self.heap_size - 1)

    def extract_min(self) -> int:
        min_elem = self.heap[0]
        self.heap[0] = self.heap[self.heap_size - 1]
        self.heap.pop(self.heap_size - 1)
        self.heap_size -= 1
        self.sift_down(0)

        return min_elem


if __name__ == '__main__':
    test_ar = [4, 5, 8, 2]
    heap = BinaryHeap(test_ar)
    print(heap())
    heap.extract_min()
    print(heap())
