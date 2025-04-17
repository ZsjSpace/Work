#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <list>
#include <mutex>
#include <thread>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "dirent.h"

// LSH哈希函数
class LSHHash {
public:
    LSHHash(int inputDim, int outputDim) : inputDim(inputDim), outputDim(outputDim) {
        // 随机初始化投影矩阵
        // 为每个输出维度生成一个随机投影向量
        for (int i = 0; i < outputDim; ++i) {
            std::vector<float> proj;
            // 每个投影向量的长度与输入维度相同
            for (int j = 0; j < inputDim; ++j) {
                // 每个元素是在[0,1]区间均匀分布的随机值
                proj.push_back(static_cast<float>(rand()) / RAND_MAX);
            }
            // 所有投影向量存储在矩阵中
            projections.push_back(proj);
        }
    }

    std::vector<int> hash(const std::vector<float>& vec) {
        std::vector<int> hashes;
        for (const auto& proj : projections) {
            float sum = 0.0f;
            for (int i = 0; i < inputDim; ++i) {
                sum += vec[i] * proj[i];
            }
            hashes.push_back(sum > 0 ? 1 : 0);
        }
        return hashes;
    }


    int inputDim; // 输入向量的维度
    int outputDim; // 输出哈希值的维度
    std::vector<std::vector<float>> projections; // 投影矩阵
};

// B+树节点基类
class BPlusNode {
public:
    virtual ~BPlusNode() = default;
    virtual bool isLeaf() const = 0;
    virtual int size() const = 0;
};

// B+树非叶子节点
class InternalNode : public BPlusNode {
public:
    InternalNode(int order) : order(order) {}

    bool isLeaf() const override { return false; }
    int size() const override { return keys.size(); }

    void insert(float key, std::shared_ptr<BPlusNode> child) {
        auto it = std::lower_bound(keys.begin(), keys.end(), key);
        int pos = it - keys.begin();
        keys.insert(it, key);
        children.insert(children.begin() + pos + 1, child);
    }

    std::shared_ptr<BPlusNode> findChild(float key) const {
        auto it = std::upper_bound(keys.begin(), keys.end(), key);
        int pos = it - keys.begin();
        return children[pos];
    }

    bool isFull() const { return keys.size() >= order - 1; }

    std::pair<float, std::shared_ptr<InternalNode>> split() {
        int mid = keys.size() / 2;
        float midKey = keys[mid];

        auto newNode = std::make_shared<InternalNode>(order);
        newNode->keys.assign(keys.begin() + mid + 1, keys.end());
        newNode->children.assign(children.begin() + mid + 1, children.end());

        keys.resize(mid);
        children.resize(mid + 1);

        return { midKey, newNode };
    }


    int order;
    std::vector<float> keys;
    std::vector<std::shared_ptr<BPlusNode>> children;
};

// B+树叶子节点
class LeafNode : public BPlusNode {
public:
    LeafNode(int order) : order(order), next(nullptr) {}

    bool isLeaf() const override { return true; }
    int size() const override { return keys.size(); }

    void insert(float key, const std::vector<float>& value) {
        auto it = std::lower_bound(keys.begin(), keys.end(), key);
        int pos = it - keys.begin();
        keys.insert(it, key);
        values.insert(values.begin() + pos, value);
    }

    bool isFull() const { return keys.size() >= order; }

    std::pair<float, std::shared_ptr<LeafNode>> split() {
        int mid = keys.size() / 2;
        float midKey = keys[mid];

        auto newNode = std::make_shared<LeafNode>(order);
        newNode->keys.assign(keys.begin() + mid, keys.end());
        newNode->values.assign(values.begin() + mid, values.end());

        keys.resize(mid);
        values.resize(mid);

        newNode->next = next;
        next = newNode;

        return { midKey, newNode };
    }

    void rangeQuery(float low, float high, std::vector<std::vector<float>>& results) const {
        auto lowIt = std::lower_bound(keys.begin(), keys.end(), low);
        auto highIt = std::upper_bound(keys.begin(), keys.end(), high);

        int start = lowIt - keys.begin();
        int end = highIt - keys.begin();

        for (int i = start; i < end; ++i) {
            results.push_back(values[i]);
        }
    }

    std::shared_ptr<LeafNode> next;


    int order;
    std::vector<float> keys;
    std::vector<std::vector<float>> values;
};

// 改进的B+树
class ImprovedBPlusTree {
public:
    ImprovedBPlusTree(int order, int inputDim, int outputDim)
        : order(order), lsh(inputDim, outputDim), root(nullptr) {
    }

    void insert(const std::vector<float>& vec) {
        auto hashes = lsh.hash(vec);
        float key = convertHashToKey(hashes);

        if (!root) {
            root = std::make_shared<LeafNode>(order);
        }

        auto [newKey, newChild] = insertRecursive(root, key, vec);

        if (newChild) {
            auto newRoot = std::make_shared<InternalNode>(order);
            newRoot->keys.push_back(newKey);
            newRoot->children.push_back(root);
            newRoot->children.push_back(newChild);
            root = newRoot;
        }
    }

    std::vector<std::vector<float>> rangeQuery(const std::vector<float>& queryVec, float radius) {
        auto hashes = lsh.hash(queryVec);
        float centerKey = convertHashToKey(hashes);
        float lowKey = centerKey - radius;
        float highKey = centerKey + radius;

        std::vector<std::vector<float>> results;
        rangeQueryRecursive(root, lowKey, highKey, results);
        return results;
    }

    void buildIndex(const std::vector<std::vector<float>>& dataset) {
        // 并行构建索引
        unsigned numThreads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;

        int chunkSize = dataset.size() / numThreads;
        for (unsigned i = 0; i < numThreads; ++i) {
            int start = i * chunkSize;
            int end = (i == numThreads - 1) ? dataset.size() : (i + 1) * chunkSize;

            threads.emplace_back([&, start, end]() {
                for (int j = start; j < end; ++j) {
                    std::lock_guard<std::mutex> lock(insertMutex);
                    insert(dataset[j]);
                }
                });
        }

        for (auto& t : threads) {
            t.join();
        }
    }


    std::pair<float, std::shared_ptr<BPlusNode>> insertRecursive(
        std::shared_ptr<BPlusNode> node, float key, const std::vector<float>& value) {

        if (node->isLeaf()) {
            auto leaf = std::static_pointer_cast<LeafNode>(node);
            leaf->insert(key, value);

            if (leaf->isFull()) {
                auto [newKey, newLeaf] = leaf->split();
                return { newKey, newLeaf };
            }
            return { 0, nullptr };
        }
        else {
            auto internal = std::static_pointer_cast<InternalNode>(node);
            auto child = internal->findChild(key);
            auto [newKey, newChild] = insertRecursive(child, key, value);

            if (newChild) {
                internal->insert(newKey, newChild);

                if (internal->isFull()) {
                    return internal->split();
                }
            }
            return { 0, nullptr };
        }
    }

    void rangeQueryRecursive(std::shared_ptr<BPlusNode> node, float low, float high,
        std::vector<std::vector<float>>& results) const {
        if (node->isLeaf()) {
            auto leaf = std::static_pointer_cast<LeafNode>(node);
            leaf->rangeQuery(low, high, results);

            // 检查后续叶子节点
            auto next = leaf->next;
            while (next && next->keys.front() <= high) {
                next->rangeQuery(low, high, results);
                next = next->next;
            }
        }
        else {
            auto internal = std::static_pointer_cast<InternalNode>(node);

            // 找到第一个key >= low的子节点
            auto it = std::upper_bound(internal->keys.begin(), internal->keys.end(), low);
            int start = it - internal->keys.begin();

            // 找到最后一个key <= high的子节点
            it = std::upper_bound(internal->keys.begin(), internal->keys.end(), high);
            int end = it - internal->keys.begin();

            for (int i = start; i <= end; ++i) {
                if (i < internal->children.size()) {
                    rangeQueryRecursive(internal->children[i], low, high, results);
                }
            }
        }
    }

    float convertHashToKey(const std::vector<int>& hashes) {
        float key = 0.0f;
        for (int i = 0; i < hashes.size(); ++i) {
            if (hashes[i]) {
                key += std::pow(2.0f, i);
            }
        }
        return key;
    }

    int order;
    LSHHash lsh;
    std::shared_ptr<BPlusNode> root;
    mutable std::mutex insertMutex;
};

// 图片特征提取类
//class FeatureExtractor {
//public:
//    FeatureExtractor() {
//        // 加载预训练模型（这里以ResNet50为例）
//        model = cv::dnn::readNetFromTensorflow("resnet50.pb", "resnet50.pbtxt");
//        if (model.empty()) {
//            throw std::runtime_error("Failed to load model");
//        }
//    }
//
//    std::vector<float> extract(const std::string& imagePath) {
//        cv::Mat img = cv::imread(imagePath);
//        if (img.empty()) {
//            throw std::runtime_error("Failed to load image: " + imagePath);
//        }
//
//        // 预处理图像
//        cv::Mat blob;
//        cv::dnn::blobFromImage(img, blob, 1.0, cv::Size(224, 224),
//            cv::Scalar(103.939, 116.779, 123.68), false, false);
//
//        // 前向传播获取特征
//        model.setInput(blob);
//        cv::Mat features = model.forward("avg_pool"); // ResNet50的全局平均池化层
//
//        // 将特征矩阵转换为向量
//        std::vector<float> featureVec;
//        featureVec.assign((float*)features.datastart, (float*)features.dataend);
//
//        return featureVec;
//    }
//
//private:
//    cv::dnn::Net model;
//};

// 缓存管理器
class CacheManager {
public:
    CacheManager(size_t capacity) : capacity(capacity) {}

    std::vector<std::vector<float>> get(float key) {
        auto it = cacheMap.find(key);
        if (it == cacheMap.end()) {
            return {};
        }

        // 移动到链表头部
        cacheList.splice(cacheList.begin(), cacheList, it->second);
        return it->second->second;
    }

    void put(float key, const std::vector<std::vector<float>>& value) {
        auto it = cacheMap.find(key);
        if (it != cacheMap.end()) {
            // 更新值并移动到头部
            it->second->second = value;
            cacheList.splice(cacheList.begin(), cacheList, it->second);
            return;
        }

        if (cacheMap.size() >= capacity) {
            // 移除最近最少使用的项
            float keyToRemove = cacheList.back().first;
            cacheMap.erase(keyToRemove);
            cacheList.pop_back();
        }

        // 插入新项
        cacheList.emplace_front(key, value);
        cacheMap[key] = cacheList.begin();
    }


    size_t capacity;
    std::list<std::pair<float, std::vector<std::vector<float>>>> cacheList;
    std::unordered_map<float,
        std::list<std::pair<float, std::vector<std::vector<float>>>>::iterator> cacheMap;
};

// 图像检索系统
class ImageRetrievalSystem {
public:
    ImageRetrievalSystem(int bptOrder, int inputDim, int lshDim, size_t cacheSize)
        : tree(bptOrder, inputDim, lshDim), cache(cacheSize), inputDim(inputDim) {
    }

    void addImage(const std::vector<float>& features) {
        if (features.size() != inputDim) {
            throw std::invalid_argument("Feature dimension mismatch");
        }
        tree.insert(features);
    }

    std::vector<std::vector<float>> searchSimilar(const std::vector<float>& query, float radius) {
        // 先检查缓存
        auto hashes = tree.lsh.hash(query);
        float key = tree.convertHashToKey(hashes);
        auto cached = cache.get(key);
        if (!cached.empty()) {
            return cached;
        }

        // 执行查询
        auto results = tree.rangeQuery(query, radius);

        // 更新缓存
        cache.put(key, results);

        return results;
    }

    void buildIndex(const std::vector<std::vector<float>>& dataset) {
        tree.buildIndex(dataset);
    }


    ImprovedBPlusTree tree;
    CacheManager cache;
    int inputDim;
};

// 图像特征提取
class FeatureExtractor {
public:
    FeatureExtractor() {
        // 加载预训练模型（这里以ResNet50为例）
        model = cv::dnn::readNetFromTensorflow("resnet50.pb", "resnet50.pbtxt");
        if (model.empty()) {
            throw std::runtime_error("Failed to load model");
        }
    }

    std::vector<float> extract(const std::string& imagePath) {
        cv::Mat img = cv::imread(imagePath);
        if (img.empty()) {
            throw std::runtime_error("Failed to load image: " + imagePath);
        }

        // 预处理图像
        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1.0, cv::Size(224, 224),
            cv::Scalar(103.939, 116.779, 123.68), false, false);

        // 前向传播获取特征
        model.setInput(blob);
        cv::Mat features = model.forward("avg_pool"); // ResNet50的全局平均池化层

        // 将特征矩阵转换为向量
        std::vector<float> featureVec;
        featureVec.assign((float*)features.datastart, (float*)features.dataend);

        return featureVec;
    }

private:
    cv::dnn::Net model;
};

// 加载本地图片数据集
std::vector<std::vector<float>> loadDataset(const std::string& datasetDir) {
    std::vector<std::vector<float>> dataset;
    FeatureExtractor extractor;

    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(datasetDir.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            std::string filename = ent->d_name;
            if (filename == "." || filename == "..") continue;

            std::string fullPath = datasetDir + "/" + filename;

            try {
                auto features = extractor.extract(fullPath);
                dataset.push_back(features);
                std::cout << "Processed: " << filename << std::endl;
            }
            catch (const std::exception& e) {
                std::cerr << "Error processing " << filename << ": " << e.what() << std::endl;
            }
        }
        closedir(dir);
    }
    else {
        throw std::runtime_error("Could not open directory: " + datasetDir);
    }

    return dataset;
}

//int main() {
//    const int FEATURE_DIM = 2048; // ResNet50特征维度
//    const int LSH_DIM = 64;
//    const int BPT_ORDER = 32;
//    const size_t CACHE_SIZE = 1000;
//
//    // 初始化系统
//    ImageRetrievalSystem system(BPT_ORDER, FEATURE_DIM, LSH_DIM, CACHE_SIZE);
//
//    // 加载数据集
//    std::string datasetDir = "path/to/your/images";
//    auto dataset = loadDataset(datasetDir);
//
//    // 构建索引
//    system.buildIndex(dataset);
//
//    // 测试查询
//    std::string queryImage = "path/to/query/image.jpg";
//    FeatureExtractor extractor;
//    auto queryFeatures = extractor.extract(queryImage);
//
//    float searchRadius = 0.2f;
//    auto results = system.searchSimilar(queryFeatures, searchRadius);
//
//    std::cout << "Found " << results.size() << " similar images." << std::endl;
//
//    return 0;
//}

int main() {
    // 示例使用
    const int FEATURE_DIM = 512; // 图像特征向量的维度
    const int LSH_DIM = 64; // LSH的输出维度
    const int BPT_ORDER = 32; // B+树的阶数
    const size_t CACHE_SIZE = 1000; // 缓存管理器容量

    // 系统初始化
    ImageRetrievalSystem system(BPT_ORDER, FEATURE_DIM, LSH_DIM, CACHE_SIZE);

    /*实际应用中，这里应该替换为从图片文件提取真实特征的代码*/
    // 模拟数据集
    std::vector<std::vector<float>> dataset;
    // 创建10000个特征向量
    for (int i = 0; i < 10000; ++i) {
        std::vector<float> features(FEATURE_DIM);
        // 为每个特征向量填充512个随机浮点数
        for (int j = 0; j < FEATURE_DIM; ++j) {
            features[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        // 将生成的向量添加到数据集中
        dataset.push_back(features);
    }

    // 构建索引
    system.buildIndex(dataset);

    // 模拟查询
    // 生成一个随机查询特征向量，模拟用户提供的查询图片特征
    std::vector<float> query(FEATURE_DIM);
    for (int j = 0; j < FEATURE_DIM; ++j) {
        query[j] = static_cast<float>(rand()) / RAND_MAX;
    }

    //  定义搜索半径
    float searchRadius = 0.2f;
    // 执行相似性搜索
    auto results = system.searchSimilar(query, searchRadius);

    std::cout << "Found " << results.size() << " similar images." << std::endl;

    return 0;
}