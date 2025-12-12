#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>

#ifdef USE_FLOAT32
    #define float_t float
#else
    #define float_t double 
#endif

#ifdef USE_INT32
    #define MY_SIZE_MAX UINT32_MAX
    #define idx_t uint32_t
#else
    #define MY_SIZE_MAX UINT64_MAX
    #define idx_t uint64_t
#endif

#define HEAP_LCH(x) (2*x + 1)
#define HEAP_RCH(x) (2*x + 2)
#define HEAP_PARENT(x) (x-1)/2  

#define HP_LEFT_SIDE 0
#define HP_RIGHT_SIDE 1

#define ALIGNMENT 64
#define CHECK_ALLOCATION_NO_CTX(x) if(!x){printf("[!!!] Failed allocation: %s at line %d \n", __FILE__, __LINE__ ); exit(1);}
#define MY_MALLOC(n) ({void* p = aligned_alloc(ALIGNMENT,n); CHECK_ALLOCATION_NO_CTX(p); memset(p, 0, n); p; })


typedef struct {
   float_t value;
   idx_t   array_idx;
} heap_node_t;

typedef struct  {
   size_t size; 
   size_t count;
   heap_node_t* data;
} heap_t;

static inline void heap_node_swap(heap_node_t* a, heap_node_t* b){
    heap_node_t tmp = *a;
    *a = *b;
    *b = tmp;
}

static void heap_allocate(heap_t* H, idx_t n){
    H -> data = (heap_node_t*)MY_MALLOC(n*sizeof(heap_node_t));
    H -> size = n;
    H -> count = 0;
    return;
}

static void heap_initialize(heap_t* H){
    for(idx_t i = 0; i < H->size; ++i)
    {
        H -> data[i].value = 0.;
        H -> data[i].array_idx = MY_SIZE_MAX; 
    }
    return;
}

static void free_heap(heap_t * H){ free(H -> data);}


// Optimized heapify (sift-down) that avoids repeated swaps.
// It correctly uses H->count as the boundary.
static inline void heapify_max_heap(heap_t* H, idx_t i) {
    heap_node_t node_to_sift = H->data[i]; // Save the node we are sifting down
    idx_t child;
    while ((child = HEAP_LCH(i)) < H->count) {
        idx_t r_child = HEAP_RCH(i);
        // Find the largest child
        
        // if (r_child < H->count && H->data[r_child].value > H->data[child].value) {
        //     child = r_child;
        // }
        
        child = (r_child < H->count && H->data[r_child].value > H->data[child].value) ? r_child : child;
        // If the node to sift is larger than its largest child, we are done
        if (node_to_sift.value >= H->data[child].value) {
            break;
        }
        // Move the larger child up
        H->data[i] = H->data[child];
        i = child;
    }
    // Place the original node in its final position
    H->data[i] = node_to_sift;
}


static void set_root_max_heap(heap_t * H, const float_t val, const idx_t array_idx){
    H -> data[0].value = val;
    H -> data[0].array_idx = array_idx;
    heapify_max_heap(H,0);
    return;
}

// Optimized insertion logic inspired by kBoundedQueue.
// It avoids swaps during bubble-up and correctly handles the full/not-full cases.
static inline void max_heap_insert(heap_t * H, const float_t val, const idx_t array_idx){
    if (H->count < H->size) {
        // Heap is not full. Add to the end and bubble up.
        idx_t i = H->count++;
        // Bubble-up without swaps
        while (i > 0) {
            idx_t parent = HEAP_PARENT(i);
            if (val <= H->data[parent].value) {
                break; // Found the correct spot
            }
            // Move parent down
            H->data[i] = H->data[parent];
            i = parent;
        }
        H->data[i].value = val;
        H->data[i].array_idx = array_idx;
    } else if (val < H->data[0].value) {
        // Heap is full, but new value is smaller than the max.
        // Replace the root and sift it down using the optimized heapify.
        H->data[0].value = val;
        H->data[0].array_idx = array_idx;
        heapify_max_heap(H, 0);
    }
    // If heap is full and new value is >= max, do nothing.
}

static int heap_node_compare(const void* a, const void* b)
{
    const heap_node_t* aa = (const heap_node_t*)a;
    const heap_node_t* bb = (const heap_node_t*)b;
    if (aa->value < bb->value) return -1;
    if (aa->value > bb->value) return 1;
    return 0;
}


static inline void heap_sort(heap_t* H){
    idx_t n = H->count;
    for(idx_t i = (H->count) - 1; i > 0; --i)
    {
        heap_node_swap(H->data, H->data + i);
        H->count = i;
        heapify_max_heap(H,0);
    }
    H->size = n;
    // qsort(H->data, H->count, sizeof(heap_node_t), heap_node_compare);
}
