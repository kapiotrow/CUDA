#include "bfs.h"

namespace bfs
{
    void readTestcaseFromFile(const fs::path &filePath, GraphCSR &graph, unsigned int &startNode)
    {
        std::ifstream file(filePath);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file: " + filePath.string());
        }

        // Read source node idx
        file >> startNode;

        // Read edges array
        int numEdges;
        file >> numEdges;
        graph.edges.resize(numEdges);
        for (int i = 0; i < numEdges; ++i)
        {
            file >> graph.edges[i];
        }

        // Read dest array
        int numDest;
        file >> numDest;
        graph.dest.resize(numDest);
        for (int i = 0; i < numDest; ++i)
        {
            file >> graph.dest[i];
        }
    }

    __global__ void kernelGlobalQueue(int *edges, int *dest, int *label, int *pFrontier, int *cFrontier, int *pFrontierTail, int *cFrontierTail)
    {
        int pTail = *pFrontierTail;

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= pTail) return;

        int v = pFrontier[tid];
        // read level of current vertex
        int vLevel = label[v];

        // traverse adjacency list of v
        int start = edges[v];
        int end = edges[v + 1];
        for (int ei = start; ei < end; ++ei)
        {
            int nei = dest[ei];

            // Try to mark neighbor as visited using atomicCAS
            // If label[nei] == -1, set to vLevel + 1
            int expected = -1;
            int newVal = vLevel + 1;
            int old = atomicCAS(&label[nei], expected, newVal);
            if (old == -1)
            {
                // we successfully claimed this neighbor: append to cFrontier
                int pos = atomicAdd(cFrontierTail, 1); // returns old value -> free index
                cFrontier[pos] = nei;
            }
            // if old != -1 someone else already visited it -> skip
        }
        
    }

    __global__ void kernelBlockQueue(int *edges, int *dest, int *label, int *pFrontier, int *cFrontier, int *pFrontierTail, int *cFrontierTail)
    {
    }

    std::vector<int> bfsOnDevice(const GraphCSR &graph, unsigned int source, BFSQueueType queueType)
    {
        int nodes = static_cast<int>(graph.edges.size() - 1);
        std::vector<int> label(nodes, -1);
        label[source] = 0;

        int *d_edges = nullptr;
        int *d_dest = nullptr;
        int *d_label = nullptr;
        int *d_pFrontier = nullptr;
        int *d_cFrontier = nullptr;
        int *d_pFrontierTail = nullptr; // pointer to single int
        int *d_cFrontierTail = nullptr; // pointer to single int


        size_t edgesBytes = graph.edges.size() * sizeof(int);
        size_t destBytes = graph.dest.size() * sizeof(int);
        size_t labelBytes = nodes * sizeof(int);
        size_t frontierBytes = nodes * sizeof(int);
        size_t intBytes = sizeof(int);

        cudaMalloc(&d_edges, edgesBytes);
        cudaMalloc(&d_dest, destBytes);
        cudaMalloc(&d_label, labelBytes);
        cudaMalloc(&d_pFrontier, frontierBytes);
        cudaMalloc(&d_cFrontier, frontierBytes);
        cudaMalloc(&d_pFrontierTail, intBytes);
        cudaMalloc(&d_cFrontierTail, intBytes);

        cudaMemcpy(d_edges, graph.edges.data(), edgesBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dest, graph.dest.data(), destBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_label, label.data(), labelBytes, cudaMemcpyHostToDevice);

        cudaMemset(d_pFrontier, 0x00, frontierBytes);
        cudaMemcpy(d_pFrontier, &source, sizeof(int), cudaMemcpyHostToDevice);

        int pTail = 1;
        int cTail = 0;
        cudaMemcpy(d_pFrontierTail, &pTail, intBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cFrontierTail, &cTail, intBytes, cudaMemcpyHostToDevice);

        while (pTail > 0)
        {
            cudaMemcpy(&pTail, d_pFrontierTail, intBytes, cudaMemcpyDeviceToHost);

            int blocks = (pTail + BLOCK_SIZE - 1) / BLOCK_SIZE;
            dim3 grid(blocks);
            dim3 block(BLOCK_SIZE);

            kernelGlobalQueue<<<grid, block>>>(d_edges, d_dest, d_label, d_pFrontier, d_cFrontier, d_pFrontierTail, d_cFrontierTail);

            cudaDeviceSynchronize();

            cudaMemcpy(&cTail, d_cFrontierTail, intBytes, cudaMemcpyDeviceToHost);

            d_pFrontier = d_cFrontier;

            cudaMemcpy(d_pFrontierTail, &cTail, intBytes, cudaMemcpyHostToDevice);
            int zero = 0;
            cudaMemcpy(d_cFrontierTail, &zero, intBytes, cudaMemcpyHostToDevice);
        }

        cudaMemcpy(label.data(), d_label, labelBytes, cudaMemcpyDeviceToHost);
        cudaFree(d_edges);
        cudaFree(d_dest);
        cudaFree(d_label);
        cudaFree(d_pFrontier);
        cudaFree(d_cFrontier);
        cudaFree(d_pFrontierTail);
        cudaFree(d_cFrontierTail);

        return label;
    }

    std::vector<int> bfsOnHost(const GraphCSR &graph, unsigned int source)
    {
        int nodes = static_cast<int>(graph.edges.size() - 1);
        std::vector<int> label(nodes, -1);
        label[source] = 0;

        std::vector<int> pFrontier;
        pFrontier.push_back(source);

        while (!pFrontier.empty())
        {
            std::vector<int> cFrontier;
            for (const auto &cVertex : pFrontier)
            {
                for (int i = graph.edges[cVertex]; i < graph.edges[cVertex + 1]; ++i)
                {
                    int neighbor = graph.dest[i];
                    if (label[neighbor] == -1)
                    {
                        label[neighbor] = label[cVertex] + 1;
                        cFrontier.push_back(neighbor);
                    }
                }
            }
            pFrontier.swap(cFrontier);
        }

        return label;
    }
} // namespace bfs
