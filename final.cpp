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

// LSH��ϣ����
class LSHHash {
public:
    LSHHash(int inputDim, int outputDim) : inputDim(inputDim), outputDim(outputDim) {
        // �����ʼ��ͶӰ����
        // Ϊÿ�����ά������һ�����ͶӰ����
        for (int i = 0; i < outputDim; ++i) {
            std::vector<float> proj;
            // ÿ��ͶӰ�����ĳ���������ά����ͬ
            for (int j = 0; j < inputDim; ++j) {
                // ÿ��Ԫ������[0,1]������ȷֲ������ֵ
                proj.push_back(static_cast<float>(rand()) / RAND_MAX);
            }
            // ����ͶӰ�����洢�ھ�����
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


    int inputDim; // ����������ά��
    int outputDim; // �����ϣֵ��ά��
    std::vector<std::vector<float>> projections; // ͶӰ����
};

// B+���ڵ����
class BPlusNode {
public:
    virtual ~BPlusNode() = default;
    virtual bool isLeaf() const = 0;
    virtual int size() const = 0;
};

// B+����Ҷ�ӽڵ�
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

// B+��Ҷ�ӽڵ�
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

// �Ľ���B+��
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
        // ���й�������
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

            // ������Ҷ�ӽڵ�
            auto next = leaf->next;
            while (next && next->keys.front() <= high) {
                next->rangeQuery(low, high, results);
                next = next->next;
            }
        }
        else {
            auto internal = std::static_pointer_cast<InternalNode>(node);

            // �ҵ���һ��key >= low���ӽڵ�
            auto it = std::upper_bound(internal->keys.begin(), internal->keys.end(), low);
            int start = it - internal->keys.begin();

            // �ҵ����һ��key <= high���ӽڵ�
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

// ͼƬ������ȡ��
//class FeatureExtractor {
//public:
//    FeatureExtractor() {
//        // ����Ԥѵ��ģ�ͣ�������ResNet50Ϊ����
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
//        // Ԥ����ͼ��
//        cv::Mat blob;
//        cv::dnn::blobFromImage(img, blob, 1.0, cv::Size(224, 224),
//            cv::Scalar(103.939, 116.779, 123.68), false, false);
//
//        // ǰ�򴫲���ȡ����
//        model.setInput(blob);
//        cv::Mat features = model.forward("avg_pool"); // ResNet50��ȫ��ƽ���ػ���
//
//        // ����������ת��Ϊ����
//        std::vector<float> featureVec;
//        featureVec.assign((float*)features.datastart, (float*)features.dataend);
//
//        return featureVec;
//    }
//
//private:
//    cv::dnn::Net model;
//};

// ���������
class CacheManager {
public:
    CacheManager(size_t capacity) : capacity(capacity) {}

    std::vector<std::vector<float>> get(float key) {
        auto it = cacheMap.find(key);
        if (it == cacheMap.end()) {
            return {};
        }

        // �ƶ�������ͷ��
        cacheList.splice(cacheList.begin(), cacheList, it->second);
        return it->second->second;
    }

    void put(float key, const std::vector<std::vector<float>>& value) {
        auto it = cacheMap.find(key);
        if (it != cacheMap.end()) {
            // ����ֵ���ƶ���ͷ��
            it->second->second = value;
            cacheList.splice(cacheList.begin(), cacheList, it->second);
            return;
        }

        if (cacheMap.size() >= capacity) {
            // �Ƴ��������ʹ�õ���
            float keyToRemove = cacheList.back().first;
            cacheMap.erase(keyToRemove);
            cacheList.pop_back();
        }

        // ��������
        cacheList.emplace_front(key, value);
        cacheMap[key] = cacheList.begin();
    }


    size_t capacity;
    std::list<std::pair<float, std::vector<std::vector<float>>>> cacheList;
    std::unordered_map<float,
        std::list<std::pair<float, std::vector<std::vector<float>>>>::iterator> cacheMap;
};

// ͼ�����ϵͳ
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
        // �ȼ�黺��
        auto hashes = tree.lsh.hash(query);
        float key = tree.convertHashToKey(hashes);
        auto cached = cache.get(key);
        if (!cached.empty()) {
            return cached;
        }

        // ִ�в�ѯ
        auto results = tree.rangeQuery(query, radius);

        // ���»���
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

// ͼ��������ȡ
class FeatureExtractor {
public:
    FeatureExtractor() {
        // ����Ԥѵ��ģ�ͣ�������ResNet50Ϊ����
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

        // Ԥ����ͼ��
        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1.0, cv::Size(224, 224),
            cv::Scalar(103.939, 116.779, 123.68), false, false);

        // ǰ�򴫲���ȡ����
        model.setInput(blob);
        cv::Mat features = model.forward("avg_pool"); // ResNet50��ȫ��ƽ���ػ���

        // ����������ת��Ϊ����
        std::vector<float> featureVec;
        featureVec.assign((float*)features.datastart, (float*)features.dataend);

        return featureVec;
    }

private:
    cv::dnn::Net model;
};

// ���ر���ͼƬ���ݼ�
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
//    const int FEATURE_DIM = 2048; // ResNet50����ά��
//    const int LSH_DIM = 64;
//    const int BPT_ORDER = 32;
//    const size_t CACHE_SIZE = 1000;
//
//    // ��ʼ��ϵͳ
//    ImageRetrievalSystem system(BPT_ORDER, FEATURE_DIM, LSH_DIM, CACHE_SIZE);
//
//    // �������ݼ�
//    std::string datasetDir = "path/to/your/images";
//    auto dataset = loadDataset(datasetDir);
//
//    // ��������
//    system.buildIndex(dataset);
//
//    // ���Բ�ѯ
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
    // ʾ��ʹ��
    const int FEATURE_DIM = 512; // ͼ������������ά��
    const int LSH_DIM = 64; // LSH�����ά��
    const int BPT_ORDER = 32; // B+���Ľ���
    const size_t CACHE_SIZE = 1000; // �������������

    // ϵͳ��ʼ��
    ImageRetrievalSystem system(BPT_ORDER, FEATURE_DIM, LSH_DIM, CACHE_SIZE);

    /*ʵ��Ӧ���У�����Ӧ���滻Ϊ��ͼƬ�ļ���ȡ��ʵ�����Ĵ���*/
    // ģ�����ݼ�
    std::vector<std::vector<float>> dataset;
    // ����10000����������
    for (int i = 0; i < 10000; ++i) {
        std::vector<float> features(FEATURE_DIM);
        // Ϊÿ�������������512�����������
        for (int j = 0; j < FEATURE_DIM; ++j) {
            features[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        // �����ɵ�������ӵ����ݼ���
        dataset.push_back(features);
    }

    // ��������
    system.buildIndex(dataset);

    // ģ���ѯ
    // ����һ�������ѯ����������ģ���û��ṩ�Ĳ�ѯͼƬ����
    std::vector<float> query(FEATURE_DIM);
    for (int j = 0; j < FEATURE_DIM; ++j) {
        query[j] = static_cast<float>(rand()) / RAND_MAX;
    }

    //  ���������뾶
    float searchRadius = 0.2f;
    // ִ������������
    auto results = system.searchSimilar(query, searchRadius);

    std::cout << "Found " << results.size() << " similar images." << std::endl;

    return 0;
}