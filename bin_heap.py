#!/usr/bin/env python3

class BinaryHeap():
    def __init__(self):
        self.heap_list = [0]
        self.size = 0

    def insert(self, x):
        self.heap_list.append(x)
        self.size += 1
        self.move_up()

    def move_up(self):
        # Kepp moving the last element by comparing it with its parent
        ptr = len(self.heap_list) - 1
        parent_ptr = ptr // 2
        while parent_ptr != 0:
            if self.heap_list[parent_ptr] > self.heap_list[ptr]:
                tmp = self.heap_list[parent_ptr]
                self.heap_list[parent_ptr] = self.heap_list[ptr]
                self.heap_list[ptr] = tmp
            else:
                break
            ptr = parent_ptr
            parent_ptr = parent_ptr // 2


    def move_down(self):
        # Kepp moving the root val by comparing it with its min. val child
        ptr = 1
        min_child_ptr = self.get_min_child_idx(ptr)
        while min_child_ptr != -1:
            if self.heap_list[ptr] > self.heap_list[min_child_ptr]:
                tmp = self.heap_list[ptr]
                self.heap_list[ptr] = self.heap_list[min_child_ptr]
                self.heap_list[min_child_ptr] = tmp
            else:
                break
            ptr = min_child_ptr
            min_child_ptr = self.get_min_child_idx(ptr)

    def get_min_child_idx(self, idx):
        child1_idx = 2 * idx
        child2_idx = 2 * idx + 1
        if child1_idx >= len(self.heap_list):
            return -1
        elif child2_idx >= len(self.heap_list):
            return child1_idx
        if self.heap_list[child1_idx] > self.heap_list[child2_idx] :
            return child2_idx
        else:
            return child1_idx



    def pop(self):
        if len(self.heap_list) == 1:
            print('Empty heap.')
            return
        ret_val = self.heap_list[1]
        self.heap_list[1] = self.heap_list[-1]
        self.heap_list.pop()
        self.size -= 1
        self.move_down()
if __name__ == '__main__':
    heap = BinaryHeap()
    heap.insert(3)
    heap.insert(6)
    heap.insert(2)
    heap.insert(5)
    heap.insert(9)
    print(heap.heap_list)
    heap.pop()
    print(heap.heap_list)
    heap.pop()
    print(heap.heap_list)
    heap.pop()
    heap.pop()
    heap.pop()
    heap.pop()


