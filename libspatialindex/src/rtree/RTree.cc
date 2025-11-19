/******************************************************************************
 * Project:  libspatialindex - A C++ library for spatial indexing
 * Author:   Marios Hadjieleftheriou, mhadji@gmail.com
 ******************************************************************************
 * Copyright (c) 2002, Marios Hadjieleftheriou
 *
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include <cstring>
#include <cmath>
#include <limits>
#include <fstream>

#include <spatialindex/SpatialIndex.h>
#include <spatialindex/capi/IdVisitor.h>
#include <spatialindex/utils/vector_utils.h>
#include "Node.h"
#include "Leaf.h"
#include "Index.h"
#include "BulkLoader.h"
#include "RTree.h"

#include "spatialindex/utils/print_utils.h"


using namespace SpatialIndex::RTree;
using namespace SpatialIndex;

SpatialIndex::RTree::Data::Data(uint32_t len, uint8_t* pData, Region& r, id_type id)
    : m_id(id), m_region(r), m_pData(nullptr), m_dataLength(len)
{
    if (m_dataLength > 0)
    {
        m_pData = new uint8_t[m_dataLength];
        memcpy(m_pData, pData, m_dataLength);
    }
}

SpatialIndex::RTree::Data::~Data()
{
    delete[] m_pData;
}

SpatialIndex::RTree::Data* SpatialIndex::RTree::Data::clone()
{
    return new Data(m_dataLength, m_pData, m_region, m_id);
}

id_type SpatialIndex::RTree::Data::getIdentifier() const
{
    return m_id;
}

void SpatialIndex::RTree::Data::getShape(IShape** out) const
{
    *out = new Region(m_region);
}

void SpatialIndex::RTree::Data::getData(uint32_t& len, uint8_t** data) const
{
    len = m_dataLength;
    *data = nullptr;

    if (m_dataLength > 0)
    {
        *data = new uint8_t[m_dataLength];
        memcpy(*data, m_pData, m_dataLength);
    }
}

uint32_t SpatialIndex::RTree::Data::getByteArraySize()
{
    return
        sizeof(id_type) +
        sizeof(uint32_t) +
        m_dataLength +
        m_region.getByteArraySize();
}

void SpatialIndex::RTree::Data::loadFromByteArray(const uint8_t* ptr)
{
    memcpy(&m_id, ptr, sizeof(id_type));
    ptr += sizeof(id_type);

    delete[] m_pData;
    m_pData = nullptr;

    memcpy(&m_dataLength, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    if (m_dataLength > 0)
    {
        m_pData = new uint8_t[m_dataLength];
        memcpy(m_pData, ptr, m_dataLength);
        ptr += m_dataLength;
    }

    m_region.loadFromByteArray(ptr);
}

void SpatialIndex::RTree::Data::storeToByteArray(uint8_t** data, uint32_t& len)
{
    // it is thread safe this way.
    uint32_t regionsize;
    uint8_t* regiondata = nullptr;
    m_region.storeToByteArray(&regiondata, regionsize);

    len = sizeof(id_type) + sizeof(uint32_t) + m_dataLength + regionsize;

    *data = new uint8_t[len];
    uint8_t* ptr = *data;

    memcpy(ptr, &m_id, sizeof(id_type));
    ptr += sizeof(id_type);
    memcpy(ptr, &m_dataLength, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    if (m_dataLength > 0)
    {
        memcpy(ptr, m_pData, m_dataLength);
        ptr += m_dataLength;
    }

    memcpy(ptr, regiondata, regionsize);
    delete[] regiondata;
    // ptr += regionsize;
}

SpatialIndex::ISpatialIndex* SpatialIndex::RTree::returnRTree(SpatialIndex::IStorageManager& sm, Tools::PropertySet& ps)
{
    SpatialIndex::ISpatialIndex* si = new SpatialIndex::RTree::RTree(sm, ps);
    return si;
}

SpatialIndex::ISpatialIndex* SpatialIndex::RTree::createNewRTree(
    SpatialIndex::IStorageManager& sm,
    double fillFactor,
    uint32_t indexCapacity,
    uint32_t leafCapacity,
    uint32_t dimension,
    RTreeVariant rv,
    id_type& indexIdentifier)
{
    Tools::Variant var;
    Tools::PropertySet ps;

    var.m_varType = Tools::VT_DOUBLE;
    var.m_val.dblVal = fillFactor;
    ps.setProperty("FillFactor", var);

    var.m_varType = Tools::VT_ULONG;
    var.m_val.ulVal = indexCapacity;
    ps.setProperty("IndexCapacity", var);

    var.m_varType = Tools::VT_ULONG;
    var.m_val.ulVal = leafCapacity;
    ps.setProperty("LeafCapacity", var);

    var.m_varType = Tools::VT_ULONG;
    var.m_val.ulVal = dimension;
    ps.setProperty("Dimension", var);

    var.m_varType = Tools::VT_LONG;
    var.m_val.lVal = rv;
    ps.setProperty("TreeVariant", var);

    ISpatialIndex* ret = returnRTree(sm, ps);

    var.m_varType = Tools::VT_LONGLONG;
    var = ps.getProperty("IndexIdentifier");
    indexIdentifier = var.m_val.llVal;

    return ret;
}

SpatialIndex::ISpatialIndex* SpatialIndex::RTree::createAndBulkLoadNewRTree(
    BulkLoadMethod m,
    IDataStream& stream,
    SpatialIndex::IStorageManager& sm,
    double fillFactor,
    uint32_t indexCapacity,
    uint32_t leafCapacity,
    uint32_t dimension,
    SpatialIndex::RTree::RTreeVariant rv,
    id_type& indexIdentifier)
{
    SpatialIndex::ISpatialIndex* tree = createNewRTree(sm, fillFactor, indexCapacity, leafCapacity, dimension, rv,
                                                       indexIdentifier);


    uint32_t bindex = static_cast<uint32_t>(std::floor(static_cast<double>(indexCapacity * fillFactor)));
    uint32_t bleaf = static_cast<uint32_t>(std::floor(static_cast<double>(leafCapacity * fillFactor)));

    SpatialIndex::RTree::BulkLoader bl;

    switch (m)
    {
    case BLM_STR:
        bl.bulkLoadUsingSTR(static_cast<RTree*>(tree), stream, bindex, bleaf, 10000, 100);
        break;
    default:
        throw Tools::IllegalArgumentException("createAndBulkLoadNewRTree: Unknown bulk load method.");
        break;
    }

    return tree;
}

SpatialIndex::ISpatialIndex* SpatialIndex::RTree::createAndBulkLoadNewRTree(
    BulkLoadMethod m,
    IDataStream& stream,
    SpatialIndex::IStorageManager& sm,
    Tools::PropertySet& ps,
    id_type& indexIdentifier)
{
    Tools::Variant var;
    RTreeVariant rv(RV_LINEAR);
    double fillFactor(0.0);
    uint32_t indexCapacity(0);
    uint32_t leafCapacity(0);
    uint32_t dimension(0);
    uint32_t pageSize(0);
    uint32_t numberOfPages(0);

    // tree variant
    var = ps.getProperty("TreeVariant");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (
            var.m_varType != Tools::VT_LONG ||
            (var.m_val.lVal != RV_LINEAR &&
                var.m_val.lVal != RV_QUADRATIC &&
                var.m_val.lVal != RV_RSTAR &&
                var.m_val.lVal != RV_BEST_SUBTREE &&
                var.m_val.lVal != RV_BEST_SPLIT &&
                var.m_val.lVal != RV_BEST_BOTH &&
                var.m_val.lVal != RV_MODEL_SUBTREE &&
                var.m_val.lVal != RV_MODEL_SPLIT &&
                var.m_val.lVal != RV_MODEL_BOTH &&
                var.m_val.lVal != RV_REF))
            throw Tools::IllegalArgumentException(
                "createAndBulkLoadNewRTree: Property TreeVariant must be Tools::VT_LONG and of RTreeVariant type");

        rv = static_cast<RTreeVariant>(var.m_val.lVal);
    }

    // fill factor
    // it cannot be larger than 50%, since linear and quadratic split algorithms
    // require assigning to both nodes the same number of entries.
    var = ps.getProperty("FillFactor");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_DOUBLE)
            throw Tools::IllegalArgumentException(
                "createAndBulkLoadNewRTree: Property FillFactor was not of type Tools::VT_DOUBLE");

        if (var.m_val.dblVal <= 0.0)
            throw Tools::IllegalArgumentException("createAndBulkLoadNewRTree: Property FillFactor was less than 0.0");

        if (((rv == RV_LINEAR || rv == RV_QUADRATIC) && var.m_val.dblVal > 0.5))
            throw Tools::IllegalArgumentException(
                "createAndBulkLoadNewRTree: Property FillFactor must be in range (0.0, 0.5) for LINEAR or QUADRATIC index types");
        if (var.m_val.dblVal >= 1.0)
            throw Tools::IllegalArgumentException(
                "createAndBulkLoadNewRTree: Property FillFactor must be in range (0.0, 1.0) for RSTAR index type");
        fillFactor = var.m_val.dblVal;
    }

    // index capacity
    var = ps.getProperty("IndexCapacity");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG || var.m_val.ulVal < 4)
            throw Tools::IllegalArgumentException(
                "createAndBulkLoadNewRTree: Property IndexCapacity must be Tools::VT_ULONG and >= 4");

        indexCapacity = var.m_val.ulVal;
    }

    // leaf capacity
    var = ps.getProperty("LeafCapacity");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG || var.m_val.ulVal < 4)
            throw Tools::IllegalArgumentException(
                "createAndBulkLoadNewRTree: Property LeafCapacity must be Tools::VT_ULONG and >= 4");

        leafCapacity = var.m_val.ulVal;
    }

    // dimension
    var = ps.getProperty("Dimension");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG)
            throw Tools::IllegalArgumentException(
                "createAndBulkLoadNewRTree: Property Dimension must be Tools::VT_ULONG");
        if (var.m_val.ulVal <= 1)
            throw Tools::IllegalArgumentException(
                "createAndBulkLoadNewRTree: Property Dimension must be greater than 1");

        dimension = var.m_val.ulVal;
    }

    // page size
    var = ps.getProperty("ExternalSortBufferPageSize");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG)
            throw Tools::IllegalArgumentException(
                "createAndBulkLoadNewRTree: Property ExternalSortBufferPageSize must be Tools::VT_ULONG");
        if (var.m_val.ulVal <= 1)
            throw Tools::IllegalArgumentException(
                "createAndBulkLoadNewRTree: Property ExternalSortBufferPageSize must be greater than 1");

        pageSize = var.m_val.ulVal;
    }

    // number of pages
    var = ps.getProperty("ExternalSortBufferTotalPages");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG)
            throw Tools::IllegalArgumentException(
                "createAndBulkLoadNewRTree: Property ExternalSortBufferTotalPages must be Tools::VT_ULONG");
        if (var.m_val.ulVal <= 1)
            throw Tools::IllegalArgumentException(
                "createAndBulkLoadNewRTree: Property ExternalSortBufferTotalPages must be greater than 1");

        numberOfPages = var.m_val.ulVal;
    }

    SpatialIndex::ISpatialIndex* tree = createNewRTree(sm, fillFactor, indexCapacity, leafCapacity, dimension, rv,
                                                       indexIdentifier);

    uint32_t bindex = static_cast<uint32_t>(std::floor(static_cast<double>(indexCapacity * fillFactor)));
    uint32_t bleaf = static_cast<uint32_t>(std::floor(static_cast<double>(leafCapacity * fillFactor)));

    SpatialIndex::RTree::BulkLoader bl;

    switch (m)
    {
    case BLM_STR:
        bl.bulkLoadUsingSTR(static_cast<RTree*>(tree), stream, bindex, bleaf, pageSize, numberOfPages);
        break;
    default:
        throw Tools::IllegalArgumentException("createAndBulkLoadNewRTree: Unknown bulk load method.");
        break;
    }

    return tree;
}

ISpatialIndex* SpatialIndex::RTree::cloneRTree(IStorageManager& sm, ISpatialIndex* refTree, RTreeVariant RV)
{
    RTree* tree = new SpatialIndex::RTree::RTree(sm, static_cast<RTree*>(refTree), RV);
    return tree;
}


SpatialIndex::ISpatialIndex* SpatialIndex::RTree::loadRTree(IStorageManager& sm, id_type indexIdentifier)
{
    Tools::Variant var;
    Tools::PropertySet ps;

    var.m_varType = Tools::VT_LONGLONG;
    var.m_val.llVal = indexIdentifier;
    ps.setProperty("IndexIdentifier", var);

    return returnRTree(sm, ps);
}

SpatialIndex::RTree::RTree::RTree(IStorageManager& sm, RTree* refTree, RTreeVariant RV):
    m_pStorageManager(&sm),
    m_rootID(StorageManager::NewPage),
    m_headerID(StorageManager::NewPage),
    m_treeVariant(RV),
    m_fillFactor(refTree->m_fillFactor),
    m_indexCapacity(refTree->m_indexCapacity),
    m_leafCapacity(refTree->m_leafCapacity),
    m_nearMinimumOverlapFactor(refTree->m_nearMinimumOverlapFactor),
    m_splitDistributionFactor(refTree->m_splitDistributionFactor),
    m_reinsertFactor(refTree->m_reinsertFactor),
    m_dimension(refTree->m_dimension),
    m_bTightMBRs(refTree->m_bTightMBRs),
    m_infiniteRegion(refTree->m_infiniteRegion),
    m_stats(refTree->m_stats),
    m_pointPool(500),
    m_regionPool(1000),
    m_indexPool(100),
    m_leafPool(100)
{
    m_indexPool.setCapacity(refTree->m_indexPool.getCapacity());
    m_leafPool.setCapacity(refTree->m_leafPool.getCapacity());
    m_regionPool.setCapacity(refTree->m_regionPool.getCapacity());
    m_pointPool.setCapacity(refTree->m_pointPool.getCapacity());
    // m_infiniteRegion.makeInfinite(m_dimension);

    NodePtr refRoot = refTree->readNode_wo_modify_stats(refTree->m_rootID);
    assert(refRoot->m_level > 0);

    Index root(this, -1, refRoot->m_level);
    // std::cout << "1. It is OK here." << std::endl;
    root.cloneFromRef(refTree, *refRoot);
    // std::cout << "2. It is OK here." << std::endl;
    m_rootID = writeNode_wo_modify_stats(&root);
    // std::cout << "3. It is OK here." << std::endl;
    storeHeader();
    // std::cout << "4. It is OK here." << std::endl;
}

SpatialIndex::RTree::RTree::RTree(IStorageManager& sm, Tools::PropertySet& ps) :
    m_pStorageManager(&sm),
    m_rootID(StorageManager::NewPage),
    m_headerID(StorageManager::NewPage),
    m_treeVariant(RV_RSTAR),
    m_fillFactor(0.7),
    m_indexCapacity(100),
    m_leafCapacity(100),
    m_nearMinimumOverlapFactor(32),
    m_splitDistributionFactor(0.4),
    m_reinsertFactor(0.3),
    m_dimension(2),
    m_bTightMBRs(true),
    m_pointPool(500),
    m_regionPool(1000),
    m_indexPool(100),
    m_leafPool(100)
{
    Tools::Variant var = ps.getProperty("IndexIdentifier");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType == Tools::VT_LONGLONG) m_headerID = var.m_val.llVal;
        else if (var.m_varType == Tools::VT_LONG) m_headerID = var.m_val.lVal;
            // for backward compatibility only.
        else throw Tools::IllegalArgumentException("RTree: Property IndexIdentifier must be Tools::VT_LONGLONG");

        initOld(ps);
    }
    else
    {
        initNew(ps);
        var.m_varType = Tools::VT_LONGLONG;
        var.m_val.llVal = m_headerID;
        ps.setProperty("IndexIdentifier", var);
    }
}

SpatialIndex::RTree::RTree::~RTree()
{
    storeHeader();
}


//
// ISpatialIndex interface
//

void SpatialIndex::RTree::RTree::insertData(uint32_t len, const uint8_t* pData, const IShape& shape, id_type id)
{
    if (shape.getDimension() != m_dimension) throw Tools::IllegalArgumentException(
        "insertData: Shape has the wrong number of dimensions.");

    // convert the shape into a Region (R-Trees index regions only; i.e., approximations of the shapes).
    RegionPtr mbr = m_regionPool.acquire();
    shape.getMBR(*mbr);

    uint8_t* buffer = nullptr;

    if (len > 0)
    {
        buffer = new uint8_t[len];
        memcpy(buffer, pData, len);
    }

    insertData_impl(len, buffer, *mbr, id);
    // the buffer is stored in the tree. Do not delete here.
}

bool SpatialIndex::RTree::RTree::deleteData(const IShape& shape, id_type id)
{
    if (shape.getDimension() != m_dimension) throw Tools::IllegalArgumentException(
        "deleteData: Shape has the wrong number of dimensions.");

    RegionPtr mbr = m_regionPool.acquire();
    shape.getMBR(*mbr);
    bool ret = deleteData_impl(*mbr, id);

    return ret;
}


void SpatialIndex::RTree::RTree::internalNodesQuery(const IShape& query, IVisitor& v)
{
    if (query.getDimension() != m_dimension) throw Tools::IllegalArgumentException(
        "containsWhatQuery: Shape has the wrong number of dimensions.");

#ifdef HAVE_PTHREAD_H
	Tools::LockGuard lock(&m_lock);
#endif

    try
    {
        std::stack<NodePtr> st;
        NodePtr root = readNode(m_rootID);
        st.push(root);

        while (!st.empty())
        {
            NodePtr n = st.top();
            st.pop();

            if (query.containsShape(n->m_nodeMBR))
            {
                IdVisitor vId = IdVisitor();
                visitSubTree(n, vId);
                const uint64_t nObj = vId.GetResultCount();
                uint64_t* obj = new uint64_t[nObj];
                std::copy(vId.GetResults().begin(), vId.GetResults().end(), obj);

                Data data = Data((uint32_t)(sizeof(uint64_t) * nObj), (uint8_t*)obj, n->m_nodeMBR, n->getIdentifier());
                v.visitData(data);
                ++(m_stats.m_u64QueryResults);
            }
            else
            {
                if (n->m_level == 0)
                {
                    for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
                    {
                        if (query.containsShape(*(n->m_ptrMBR[cChild])))
                        {
                            Data data = Data(sizeof(id_type), (uint8_t*)&n->m_pIdentifier[cChild],
                                             *(n->m_ptrMBR[cChild]), n->getIdentifier());
                            v.visitData(data);
                            ++(m_stats.m_u64QueryResults);
                        }
                    }
                }
                else //not a leaf
                {
                    if (query.intersectsShape(n->m_nodeMBR))
                    {
                        for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
                        {
                            st.push(readNode(n->m_pIdentifier[cChild]));
                        }
                    }
                }
            }
        }
    }
    catch (...)
    {
        throw;
    }
}

void SpatialIndex::RTree::RTree::containsWhatQuery(const IShape& query, IVisitor& v)
{
    if (query.getDimension() != m_dimension) throw Tools::IllegalArgumentException(
        "containsWhatQuery: Shape has the wrong number of dimensions.");

    try
    {
        std::stack<NodePtr> st;
        NodePtr root = readNode(m_rootID);
        st.push(root);

        while (!st.empty())
        {
            NodePtr n = st.top();
            st.pop();

            if (n->m_level == 0)
            {
                v.visitNode(*n);

                for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
                {
                    if (query.containsShape(*(n->m_ptrMBR[cChild])))
                    {
                        Data data = Data(n->m_pDataLength[cChild], n->m_pData[cChild], *(n->m_ptrMBR[cChild]),
                                         n->m_pIdentifier[cChild]);
                        v.visitData(data);
                        ++(m_stats.m_u64QueryResults);
                    }
                }
            }
            else //not a leaf
            {
                if (query.containsShape(n->m_nodeMBR))
                {
                    visitSubTree(n, v);
                }
                else if (query.intersectsShape(n->m_nodeMBR))
                {
                    v.visitNode(*n);

                    for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
                    {
                        st.push(readNode(n->m_pIdentifier[cChild]));
                    }
                }
            }
        }
    }
    catch (...)
    {
        throw;
    }
}

void SpatialIndex::RTree::RTree::intersectsWithQuery(const IShape& query, IVisitor& v)
{
    if (query.getDimension() != m_dimension) throw Tools::IllegalArgumentException(
        "intersectsWithQuery: Shape has the wrong number of dimensions.");
    rangeQuery(IntersectionQuery, query, v);
}

uint64_t SpatialIndex::RTree::RTree::intersectsWithQueryCount(const IShape& query)
{
    if (query.getDimension() != m_dimension) throw Tools::IllegalArgumentException(
        "intersectsWithQueryCount: Shape has the wrong number of dimensions.");
    return rangeQueryCount(IntersectionQuery, query);
}

uint64_t SpatialIndex::RTree::RTree::getIntersectsWithQueryIOCost(const IShape& query)
{
    if (query.getDimension() != m_dimension) throw Tools::IllegalArgumentException(
        "getIntersectsWithQueryIOCost: Shape has the wrong number of dimensions.");
    return rangeQueryIOCost(IntersectionQuery, query);
}

void SpatialIndex::RTree::RTree::pointLocationQuery(const Point& query, IVisitor& v)
{
    if (query.m_dimension != m_dimension) throw Tools::IllegalArgumentException(
        "pointLocationQuery: Shape has the wrong number of dimensions.");
    Region r(query, query);
    rangeQuery(IntersectionQuery, r, v);
}

void SpatialIndex::RTree::RTree::nearestNeighborQuery(uint32_t k, const IShape& query, IVisitor& v,
                                                      INearestNeighborComparator& nnc)
{
    if (query.getDimension() != m_dimension) throw Tools::IllegalArgumentException(
        "nearestNeighborQuery: Shape has the wrong number of dimensions.");

    auto ascending = [](const NNEntry* lhs, const NNEntry* rhs) { return lhs->m_minDist > rhs->m_minDist; };
    std::priority_queue<NNEntry*, std::vector<NNEntry*>, decltype(ascending)> queue(ascending);

    queue.push(new NNEntry(m_rootID, nullptr, 0.0));

    uint32_t count = 0;
    double knearest = 0.0;

    while (!queue.empty())
    {
        NNEntry* pFirst = queue.top();

        // report all nearest neighbors with equal greatest distances.
        // (neighbors can be more than k, if many happen to have the same greatest distance).
        if (count >= k && pFirst->m_minDist > knearest) break;

        queue.pop();

        if (pFirst->m_pEntry == nullptr)
        {
            // n is a leaf or an index.
            NodePtr n = readNode(pFirst->m_id);
            v.visitNode(*n);

            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                if (n->m_level == 0)
                {
                    Data* e = new Data(n->m_pDataLength[cChild], n->m_pData[cChild], *(n->m_ptrMBR[cChild]),
                                       n->m_pIdentifier[cChild]);
                    // we need to compare the query with the actual data entry here, so we call the
                    // appropriate getMinimumDistance method of NearestNeighborComparator.
                    queue.push(new NNEntry(n->m_pIdentifier[cChild], e, nnc.getMinimumDistance(query, *e)));
                }
                else
                {
                    queue.push(new NNEntry(n->m_pIdentifier[cChild], nullptr,
                                           nnc.getMinimumDistance(query, *(n->m_ptrMBR[cChild]))));
                }
            }
        }
        else
        {
            v.visitData(*(static_cast<IData*>(pFirst->m_pEntry)));
            ++(m_stats.m_u64QueryResults);
            ++count;
            knearest = pFirst->m_minDist;
            delete pFirst->m_pEntry;
        }

        delete pFirst;
    }

    while (!queue.empty())
    {
        NNEntry* e = queue.top();
        queue.pop();
        if (e->m_pEntry != nullptr) delete e->m_pEntry;
        delete e;
    }
}

void SpatialIndex::RTree::RTree::nearestNeighborQuery(uint32_t k, const IShape& query, IVisitor& v)
{
    if (query.getDimension() != m_dimension) throw Tools::IllegalArgumentException(
        "nearestNeighborQuery: Shape has the wrong number of dimensions.");
    NNComparator nnc;
    nearestNeighborQuery(k, query, v, nnc);
}


uint64_t SpatialIndex::RTree::RTree::getKNNQueryIOCost(uint32_t k, const IShape& query)
{
    uint64_t io_cost = 0;
    NNComparator nnc;
    if (query.getDimension() != m_dimension) throw Tools::IllegalArgumentException(
        "nearestNeighborQuery: Shape has the wrong number of dimensions.");

    auto ascending = [](const NNEntry* lhs, const NNEntry* rhs) { return lhs->m_minDist > rhs->m_minDist; };
    std::priority_queue<NNEntry*, std::vector<NNEntry*>, decltype(ascending)> queue(ascending);

    queue.push(new NNEntry(m_rootID, nullptr, 0.0));

    uint32_t count = 0;
    double knearest = 0.0;

    while (!queue.empty())
    {
        NNEntry* pFirst = queue.top();

        // report all nearest neighbors with equal greatest distances.
        // (neighbors can be more than k, if many happen to have the same greatest distance).
        if (count >= k && pFirst->m_minDist > knearest) break;

        queue.pop();

        if (pFirst->m_pEntry == nullptr)
        {
            // n is a leaf or an index.
            NodePtr n = readNode(pFirst->m_id);
            ++io_cost;

            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                if (n->m_level == 0)
                {
                    Data* e = new Data(n->m_pDataLength[cChild], n->m_pData[cChild], *(n->m_ptrMBR[cChild]),
                                       n->m_pIdentifier[cChild]);
                    // we need to compare the query with the actual data entry here, so we call the
                    // appropriate getMinimumDistance method of NearestNeighborComparator.
                    queue.push(new NNEntry(n->m_pIdentifier[cChild], e, nnc.getMinimumDistance(query, *e)));
                }
                else
                {
                    queue.push(new NNEntry(n->m_pIdentifier[cChild], nullptr,
                                           nnc.getMinimumDistance(query, *(n->m_ptrMBR[cChild]))));
                }
            }
        }
        else
        {
            ++(m_stats.m_u64QueryResults);
            ++count;
            knearest = pFirst->m_minDist;
            delete pFirst->m_pEntry;
        }

        delete pFirst;
    }

    while (!queue.empty())
    {
        NNEntry* e = queue.top();
        queue.pop();
        if (e->m_pEntry != nullptr) delete e->m_pEntry;
        delete e;
    }
    return io_cost;
}


void SpatialIndex::RTree::RTree::selfJoinQuery(const IShape& query, IVisitor& v)
{
    if (query.getDimension() != m_dimension)
        throw Tools::IllegalArgumentException("selfJoinQuery: Shape has the wrong number of dimensions.");

    RegionPtr mbr = m_regionPool.acquire();
    query.getMBR(*mbr);
    selfJoinQuery(m_rootID, m_rootID, *mbr, v);
}


void SpatialIndex::RTree::RTree::queryStrategy(IQueryStrategy& qs)
{
    id_type next = m_rootID;
    bool hasNext = true;

    while (hasNext)
    {
        NodePtr n = readNode(next);
        qs.getNextEntry(*n, next, hasNext);
    }
}

void SpatialIndex::RTree::RTree::getIndexProperties(Tools::PropertySet& out) const
{
    Tools::Variant var;

    // dimension
    var.m_varType = Tools::VT_ULONG;
    var.m_val.ulVal = m_dimension;
    out.setProperty("Dimension", var);

    // index capacity
    var.m_varType = Tools::VT_ULONG;
    var.m_val.ulVal = m_indexCapacity;
    out.setProperty("IndexCapacity", var);

    // leaf capacity
    var.m_varType = Tools::VT_ULONG;
    var.m_val.ulVal = m_leafCapacity;
    out.setProperty("LeafCapacity", var);

    // R-tree variant
    var.m_varType = Tools::VT_LONG;
    var.m_val.lVal = m_treeVariant;
    out.setProperty("TreeVariant", var);

    // fill factor
    var.m_varType = Tools::VT_DOUBLE;
    var.m_val.dblVal = m_fillFactor;
    out.setProperty("FillFactor", var);

    // near minimum overlap factor
    var.m_varType = Tools::VT_ULONG;
    var.m_val.ulVal = m_nearMinimumOverlapFactor;
    out.setProperty("NearMinimumOverlapFactor", var);

    // split distribution factor
    var.m_varType = Tools::VT_DOUBLE;
    var.m_val.dblVal = m_splitDistributionFactor;
    out.setProperty("SplitDistributionFactor", var);

    // reinsert factor
    var.m_varType = Tools::VT_DOUBLE;
    var.m_val.dblVal = m_reinsertFactor;
    out.setProperty("ReinsertFactor", var);

    // tight MBRs
    var.m_varType = Tools::VT_BOOL;
    var.m_val.blVal = m_bTightMBRs;
    out.setProperty("EnsureTightMBRs", var);

    // index pool capacity
    var.m_varType = Tools::VT_ULONG;
    var.m_val.ulVal = m_indexPool.getCapacity();
    out.setProperty("IndexPoolCapacity", var);

    // leaf pool capacity
    var.m_varType = Tools::VT_ULONG;
    var.m_val.ulVal = m_leafPool.getCapacity();
    out.setProperty("LeafPoolCapacity", var);

    // region pool capacity
    var.m_varType = Tools::VT_ULONG;
    var.m_val.ulVal = m_regionPool.getCapacity();
    out.setProperty("RegionPoolCapacity", var);

    // point pool capacity
    var.m_varType = Tools::VT_ULONG;
    var.m_val.ulVal = m_pointPool.getCapacity();
    out.setProperty("PointPoolCapacity", var);

    var.m_varType = Tools::VT_LONGLONG;
    var.m_val.llVal = m_headerID;
    out.setProperty("IndexIdentifier", var);
}

void SpatialIndex::RTree::RTree::addCommand(ICommand* pCommand, CommandType ct)
{
    switch (ct)
    {
    case CT_NODEREAD:
        m_readNodeCommands.push_back(std::shared_ptr<ICommand>(pCommand));
        break;
    case CT_NODEWRITE:
        m_writeNodeCommands.push_back(std::shared_ptr<ICommand>(pCommand));
        break;
    case CT_NODEDELETE:
        m_deleteNodeCommands.push_back(std::shared_ptr<ICommand>(pCommand));
        break;
    }
}

bool SpatialIndex::RTree::RTree::isIndexValid()
{
    bool ret = true;
    std::stack<ValidateEntry> st;
    NodePtr root = readNode(m_rootID);

    if (root->m_level != m_stats.m_u32TreeHeight - 1)
    {
        std::cerr << "Invalid tree height." << std::endl;
        return false;
    }

    std::map<uint32_t, uint32_t> nodesInLevel;
    nodesInLevel.insert(std::pair<uint32_t, uint32_t>(root->m_level, 1));

    ValidateEntry e(root->m_nodeMBR, root);
    st.push(e);

    while (!st.empty())
    {
        e = st.top();
        st.pop();

        Region tmpRegion;
        tmpRegion = m_infiniteRegion;

        for (uint32_t cDim = 0; cDim < tmpRegion.m_dimension; ++cDim)
        {
            tmpRegion.m_pLow[cDim] = std::numeric_limits<double>::max();
            tmpRegion.m_pHigh[cDim] = -std::numeric_limits<double>::max();

            for (uint32_t cChild = 0; cChild < e.m_pNode->m_children; ++cChild)
            {
                tmpRegion.m_pLow[cDim] = std::min(tmpRegion.m_pLow[cDim], e.m_pNode->m_ptrMBR[cChild]->m_pLow[cDim]);
                tmpRegion.m_pHigh[cDim] = std::max(tmpRegion.m_pHigh[cDim], e.m_pNode->m_ptrMBR[cChild]->m_pHigh[cDim]);
            }
        }

        if (!(tmpRegion == e.m_pNode->m_nodeMBR))
        {
            std::cerr << "Invalid parent information." << std::endl;
            std::cerr << "(";
            for (uint32_t cDim = 0; cDim < tmpRegion.m_dimension; ++cDim)
            {
                std::cerr << tmpRegion.m_pLow[cDim] << "," << tmpRegion.m_pHigh[cDim] << ", ";
            }
            std::cerr << ")" << std::endl;
            std::cerr << "(";
            for (uint32_t cDim = 0; cDim < tmpRegion.m_dimension; ++cDim)
            {
                std::cerr << (e.m_pNode->m_nodeMBR).m_pLow[cDim] << "," << (e.m_pNode->m_nodeMBR).m_pHigh[cDim] << ", ";
            }
            std::cerr << ")" << std::endl;

            // std::cerr << "Invalid parent information." << std::endl;
            ret = false;
        }
        else if (!(tmpRegion == e.m_parentMBR))
        {
            std::cerr << "Error in parent." << std::endl;
            ret = false;
        }

        if (e.m_pNode->m_level != 0)
        {
            for (uint32_t cChild = 0; cChild < e.m_pNode->m_children; ++cChild)
            {
                NodePtr ptrN = readNode(e.m_pNode->m_pIdentifier[cChild]);
                ValidateEntry tmpEntry(*(e.m_pNode->m_ptrMBR[cChild]), ptrN);

                std::map<uint32_t, uint32_t>::iterator itNodes = nodesInLevel.find(tmpEntry.m_pNode->m_level);

                if (itNodes == nodesInLevel.end())
                {
                    nodesInLevel.insert(std::pair<uint32_t, uint32_t>(tmpEntry.m_pNode->m_level, 1l));
                }
                else
                {
                    nodesInLevel[tmpEntry.m_pNode->m_level] = nodesInLevel[tmpEntry.m_pNode->m_level] + 1;
                }

                st.push(tmpEntry);
            }
        }
    }

    uint32_t nodes = 0;
    for (uint32_t cLevel = 0; cLevel < m_stats.m_u32TreeHeight; ++cLevel)
    {
        if (nodesInLevel[cLevel] != m_stats.m_nodesInLevel[cLevel])
        {
            std::cerr << "Invalid nodesInLevel information." << std::endl;
            ret = false;
        }

        nodes += m_stats.m_nodesInLevel[cLevel];
    }

    if (nodes != m_stats.m_u32Nodes)
    {
        std::cerr << "Invalid number of nodes information." << std::endl;
        ret = false;
    }

    return ret;
}

void SpatialIndex::RTree::RTree::getStatistics(IStatistics** out) const
{
    *out = new Statistics(m_stats);
}

void SpatialIndex::RTree::RTree::flush()
{
    storeHeader();
}


void SpatialIndex::RTree::RTree::initNew(Tools::PropertySet& ps)
{
    Tools::Variant var;

    // tree variant
    var = ps.getProperty("TreeVariant");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (
            var.m_varType != Tools::VT_LONG ||
            (var.m_val.lVal != RV_LINEAR &&
                var.m_val.lVal != RV_QUADRATIC &&
                var.m_val.lVal != RV_RSTAR &&
                var.m_val.lVal != RV_BEST_SUBTREE &&
                var.m_val.lVal != RV_BEST_SPLIT &&
                var.m_val.lVal != RV_BEST_BOTH &&
                var.m_val.lVal != RV_MODEL_SUBTREE &&
                var.m_val.lVal != RV_MODEL_SPLIT &&
                var.m_val.lVal != RV_MODEL_BOTH &&
                var.m_val.lVal != RV_REF))
            throw Tools::IllegalArgumentException(
                "initNew: Property TreeVariant must be Tools::VT_LONG and of RTreeVariant type");

        m_treeVariant = static_cast<RTreeVariant>(var.m_val.lVal);
    }

    // fill factor
    // it cannot be larger than 50%, since linear and quadratic split algorithms
    // require assigning to both nodes the same number of entries.
    var = ps.getProperty("FillFactor");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_DOUBLE)
            throw Tools::IllegalArgumentException("initNew: Property FillFactor was not of type Tools::VT_DOUBLE");

        if (var.m_val.dblVal <= 0.0)
            throw Tools::IllegalArgumentException("initNew: Property FillFactor was less than 0.0");

        if (((m_treeVariant == RV_LINEAR || m_treeVariant == RV_QUADRATIC) && var.m_val.dblVal > 0.5))
            throw Tools::IllegalArgumentException("initNew: Property FillFactor must be in range "
                "(0.0, 0.5) for LINEAR or QUADRATIC index types");
        if (var.m_val.dblVal >= 1.0)
            throw Tools::IllegalArgumentException("initNew: Property FillFactor must be in range "
                "(0.0, 1.0) for RSTAR index type");
        m_fillFactor = var.m_val.dblVal;
    }

    // index capacity
    var = ps.getProperty("IndexCapacity");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG || var.m_val.ulVal < 4)
            throw Tools::IllegalArgumentException("initNew: Property IndexCapacity must be Tools::VT_ULONG and >= 4");

        m_indexCapacity = var.m_val.ulVal;
    }

    // leaf capacity
    var = ps.getProperty("LeafCapacity");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG || var.m_val.ulVal < 4)
            throw Tools::IllegalArgumentException("initNew: Property LeafCapacity must be Tools::VT_ULONG and >= 4");

        m_leafCapacity = var.m_val.ulVal;
    }

    // near minimum overlap factor
    var = ps.getProperty("NearMinimumOverlapFactor");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (
            var.m_varType != Tools::VT_ULONG ||
            var.m_val.ulVal < 1 ||
            var.m_val.ulVal > m_indexCapacity ||
            var.m_val.ulVal > m_leafCapacity)
            throw Tools::IllegalArgumentException(
                "initNew: Property NearMinimumOverlapFactor must be Tools::VT_ULONG and less than both index and leaf capacities");

        m_nearMinimumOverlapFactor = var.m_val.ulVal;
    }

    // split distribution factor
    var = ps.getProperty("SplitDistributionFactor");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (
            var.m_varType != Tools::VT_DOUBLE ||
            var.m_val.dblVal <= 0.0 ||
            var.m_val.dblVal >= 1.0)
            throw Tools::IllegalArgumentException(
                "initNew: Property SplitDistributionFactor must be Tools::VT_DOUBLE and in (0.0, 1.0)");

        m_splitDistributionFactor = var.m_val.dblVal;
    }

    // reinsert factor
    var = ps.getProperty("ReinsertFactor");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (
            var.m_varType != Tools::VT_DOUBLE ||
            var.m_val.dblVal <= 0.0 ||
            var.m_val.dblVal >= 1.0)
            throw Tools::IllegalArgumentException(
                "initNew: Property ReinsertFactor must be Tools::VT_DOUBLE and in (0.0, 1.0)");

        m_reinsertFactor = var.m_val.dblVal;
    }

    // dimension
    var = ps.getProperty("Dimension");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG)
            throw Tools::IllegalArgumentException("initNew: Property Dimension must be Tools::VT_ULONG");
        if (var.m_val.ulVal <= 1)
            throw Tools::IllegalArgumentException("initNew: Property Dimension must be greater than 1");

        m_dimension = var.m_val.ulVal;
    }

    // tight MBRs
    var = ps.getProperty("EnsureTightMBRs");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_BOOL)
            throw Tools::IllegalArgumentException("initNew: Property EnsureTightMBRs must be Tools::VT_BOOL");

        m_bTightMBRs = var.m_val.blVal;
    }

    // index pool capacity
    var = ps.getProperty("IndexPoolCapacity");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG)
            throw Tools::IllegalArgumentException("initNew: Property IndexPoolCapacity must be Tools::VT_ULONG");

        m_indexPool.setCapacity(var.m_val.ulVal);
    }

    // leaf pool capacity
    var = ps.getProperty("LeafPoolCapacity");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG)
            throw Tools::IllegalArgumentException("initNew: Property LeafPoolCapacity must be Tools::VT_ULONG");

        m_leafPool.setCapacity(var.m_val.ulVal);
    }

    // region pool capacity
    var = ps.getProperty("RegionPoolCapacity");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG)
            throw Tools::IllegalArgumentException("initNew: Property RegionPoolCapacity must be Tools::VT_ULONG");

        m_regionPool.setCapacity(var.m_val.ulVal);
    }

    // point pool capacity
    var = ps.getProperty("PointPoolCapacity");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG)
            throw Tools::IllegalArgumentException("initNew: Property PointPoolCapacity must be Tools::VT_ULONG");

        m_pointPool.setCapacity(var.m_val.ulVal);
    }

    m_infiniteRegion.makeInfinite(m_dimension);

    m_stats.m_u32TreeHeight = 1;
    m_stats.m_nodesInLevel.push_back(0);

    Leaf root(this, -1);
    m_rootID = writeNode(&root);

    storeHeader();
}

void SpatialIndex::RTree::RTree::initOld(Tools::PropertySet& ps)
{
    loadHeader();

    // only some of the properties may be changed.
    // the rest are just ignored.

    Tools::Variant var;

    // tree variant
    var = ps.getProperty("TreeVariant");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (
            var.m_varType != Tools::VT_LONG ||
            (var.m_val.lVal != RV_LINEAR &&
                var.m_val.lVal != RV_QUADRATIC &&
                var.m_val.lVal != RV_RSTAR &&
                var.m_val.lVal != RV_BEST_SUBTREE &&
                var.m_val.lVal != RV_BEST_SPLIT &&
                var.m_val.lVal != RV_BEST_BOTH &&
                var.m_val.lVal != RV_MODEL_SUBTREE &&
                var.m_val.lVal != RV_MODEL_SPLIT &&
                var.m_val.lVal != RV_MODEL_BOTH &&
                var.m_val.lVal != RV_REF))
            throw Tools::IllegalArgumentException(
                "initOld: Property TreeVariant must be Tools::VT_LONG and of RTreeVariant type");

        m_treeVariant = static_cast<RTreeVariant>(var.m_val.lVal);
    }

    // near minimum overlap factor
    var = ps.getProperty("NearMinimumOverlapFactor");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (
            var.m_varType != Tools::VT_ULONG ||
            var.m_val.ulVal < 1 ||
            var.m_val.ulVal > m_indexCapacity ||
            var.m_val.ulVal > m_leafCapacity)
            throw Tools::IllegalArgumentException(
                "initOld: Property NearMinimumOverlapFactor must be Tools::VT_ULONG and less than both index and leaf capacities");

        m_nearMinimumOverlapFactor = var.m_val.ulVal;
    }

    // split distribution factor
    var = ps.getProperty("SplitDistributionFactor");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_DOUBLE || var.m_val.dblVal <= 0.0 || var.m_val.dblVal >= 1.0)
            throw Tools::IllegalArgumentException(
                "initOld: Property SplitDistributionFactor must be Tools::VT_DOUBLE and in (0.0, 1.0)");

        m_splitDistributionFactor = var.m_val.dblVal;
    }

    // reinsert factor
    var = ps.getProperty("ReinsertFactor");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_DOUBLE || var.m_val.dblVal <= 0.0 || var.m_val.dblVal >= 1.0)
            throw Tools::IllegalArgumentException(
                "initOld: Property ReinsertFactor must be Tools::VT_DOUBLE and in (0.0, 1.0)");

        m_reinsertFactor = var.m_val.dblVal;
    }

    // tight MBRs
    var = ps.getProperty("EnsureTightMBRs");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_BOOL) throw Tools::IllegalArgumentException(
            "initOld: Property EnsureTightMBRs must be Tools::VT_BOOL");

        m_bTightMBRs = var.m_val.blVal;
    }

    // index pool capacity
    var = ps.getProperty("IndexPoolCapacity");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG) throw Tools::IllegalArgumentException(
            "initOld: Property IndexPoolCapacity must be Tools::VT_ULONG");

        m_indexPool.setCapacity(var.m_val.ulVal);
    }

    // leaf pool capacity
    var = ps.getProperty("LeafPoolCapacity");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG) throw Tools::IllegalArgumentException(
            "initOld: Property LeafPoolCapacity must be Tools::VT_ULONG");

        m_leafPool.setCapacity(var.m_val.ulVal);
    }

    // region pool capacity
    var = ps.getProperty("RegionPoolCapacity");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG) throw Tools::IllegalArgumentException(
            "initOld: Property RegionPoolCapacity must be Tools::VT_ULONG");

        m_regionPool.setCapacity(var.m_val.ulVal);
    }

    // point pool capacity
    var = ps.getProperty("PointPoolCapacity");
    if (var.m_varType != Tools::VT_EMPTY)
    {
        if (var.m_varType != Tools::VT_ULONG) throw Tools::IllegalArgumentException(
            "initOld: Property PointPoolCapacity must be Tools::VT_ULONG");

        m_pointPool.setCapacity(var.m_val.ulVal);
    }

    m_infiniteRegion.makeInfinite(m_dimension);
}

void SpatialIndex::RTree::RTree::storeHeader()
{
    const uint32_t headerSize =
        sizeof(id_type) + // m_rootID
        sizeof(RTreeVariant) + // m_treeVariant
        sizeof(double) + // m_fillFactor
        sizeof(uint32_t) + // m_indexCapacity
        sizeof(uint32_t) + // m_leafCapacity
        sizeof(uint32_t) + // m_nearMinimumOverlapFactor
        sizeof(double) + // m_splitDistributionFactor
        sizeof(double) + // m_reinsertFactor
        sizeof(uint32_t) + // m_dimension
        sizeof(char) + // m_bTightMBRs
        sizeof(uint32_t) + // m_stats.m_nodes
        sizeof(uint64_t) + // m_stats.m_data
        sizeof(uint32_t) + // m_stats.m_treeHeight
        m_stats.m_u32TreeHeight * sizeof(uint32_t); // m_stats.m_nodesInLevel

    uint8_t* header = new uint8_t[headerSize];
    uint8_t* ptr = header;

    memcpy(ptr, &m_rootID, sizeof(id_type));
    ptr += sizeof(id_type);
    memcpy(ptr, &m_treeVariant, sizeof(RTreeVariant));
    ptr += sizeof(RTreeVariant);
    memcpy(ptr, &m_fillFactor, sizeof(double));
    ptr += sizeof(double);
    memcpy(ptr, &m_indexCapacity, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(ptr, &m_leafCapacity, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(ptr, &m_nearMinimumOverlapFactor, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(ptr, &m_splitDistributionFactor, sizeof(double));
    ptr += sizeof(double);
    memcpy(ptr, &m_reinsertFactor, sizeof(double));
    ptr += sizeof(double);
    memcpy(ptr, &m_dimension, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    char c = (char)m_bTightMBRs;
    memcpy(ptr, &c, sizeof(char));
    ptr += sizeof(char);
    memcpy(ptr, &(m_stats.m_u32Nodes), sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(ptr, &(m_stats.m_u64Data), sizeof(uint64_t));
    ptr += sizeof(uint64_t);
    memcpy(ptr, &(m_stats.m_u32TreeHeight), sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    for (uint32_t cLevel = 0; cLevel < m_stats.m_u32TreeHeight; ++cLevel)
    {
        memcpy(ptr, &(m_stats.m_nodesInLevel[cLevel]), sizeof(uint32_t));
        ptr += sizeof(uint32_t);
    }

    m_pStorageManager->storeByteArray(m_headerID, headerSize, header);

    delete[] header;
}

void SpatialIndex::RTree::RTree::loadHeader()
{
    uint32_t headerSize;
    uint8_t* header = nullptr;
    m_pStorageManager->loadByteArray(m_headerID, headerSize, &header);

    uint8_t* ptr = header;

    memcpy(&m_rootID, ptr, sizeof(id_type));
    ptr += sizeof(id_type);
    memcpy(&m_treeVariant, ptr, sizeof(RTreeVariant));
    ptr += sizeof(RTreeVariant);
    memcpy(&m_fillFactor, ptr, sizeof(double));
    ptr += sizeof(double);
    memcpy(&m_indexCapacity, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(&m_leafCapacity, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(&m_nearMinimumOverlapFactor, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(&m_splitDistributionFactor, ptr, sizeof(double));
    ptr += sizeof(double);
    memcpy(&m_reinsertFactor, ptr, sizeof(double));
    ptr += sizeof(double);
    memcpy(&m_dimension, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    char c;
    memcpy(&c, ptr, sizeof(char));
    m_bTightMBRs = (c != 0);
    ptr += sizeof(char);
    memcpy(&(m_stats.m_u32Nodes), ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(&(m_stats.m_u64Data), ptr, sizeof(uint64_t));
    ptr += sizeof(uint64_t);
    memcpy(&(m_stats.m_u32TreeHeight), ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    for (uint32_t cLevel = 0; cLevel < m_stats.m_u32TreeHeight; ++cLevel)
    {
        uint32_t cNodes;
        memcpy(&cNodes, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        m_stats.m_nodesInLevel.push_back(cNodes);
    }

    delete[] header;
}

void SpatialIndex::RTree::RTree::insertData_impl(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id)
{
    assert(mbr.getDimension() == m_dimension);

    std::stack<id_type> pathBuffer;
    uint8_t* overflowTable = nullptr;

    try
    {
        NodePtr root = readNode(m_rootID);

        overflowTable = new uint8_t[root->m_level];
        memset(overflowTable, 0, root->m_level);

        NodePtr l = root->chooseSubtree(mbr, 0, pathBuffer);
        if (l.get() == root.get())
        {
            assert(root.unique());
            root.relinquish();
        }
        l->insertData(dataLength, pData, mbr, id, pathBuffer, overflowTable);

        delete[] overflowTable;
        ++(m_stats.m_u64Data);
    }
    catch (...)
    {
        delete[] overflowTable;
        throw;
    }
}

void SpatialIndex::RTree::RTree::insertData_impl(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id,
                                                 uint32_t level, uint8_t* overflowTable)
{
    assert(mbr.getDimension() == m_dimension);

    std::stack<id_type> pathBuffer;
    NodePtr root = readNode(m_rootID);
    NodePtr n = root->chooseSubtree(mbr, level, pathBuffer);

    assert(n->m_level == level);

    if (n.get() == root.get())
    {
        assert(root.unique());
        root.relinquish();
    }
    n->insertData(dataLength, pData, mbr, id, pathBuffer, overflowTable);
}

bool SpatialIndex::RTree::RTree::deleteData_impl(const Region& mbr, id_type id)
{
    assert(mbr.m_dimension == m_dimension);

    std::stack<id_type> pathBuffer;
    NodePtr root = readNode(m_rootID);
    NodePtr l = root->findLeaf(mbr, id, pathBuffer);
    if (l.get() == root.get())
    {
        assert(root.unique());
        root.relinquish();
    }

    if (l.get() != nullptr)
    {
        Leaf* pL = static_cast<Leaf*>(l.get());
        pL->deleteData(mbr, id, pathBuffer);
        --(m_stats.m_u64Data);
        return true;
    }

    return false;
}

SpatialIndex::id_type SpatialIndex::RTree::RTree::writeNode(Node* n)
{
    uint8_t* buffer;
    uint32_t dataLength;
    n->storeToByteArray(&buffer, dataLength);

    id_type page;
    if (n->m_identifier < 0) page = StorageManager::NewPage;
    else page = n->m_identifier;

    try
    {
        m_pStorageManager->storeByteArray(page, dataLength, buffer);
        delete[] buffer;
    }
    catch (InvalidPageException& e)
    {
        delete[] buffer;
        std::cerr << e.what() << std::endl;
        throw;
    }

    if (n->m_identifier < 0)
    {
        n->m_identifier = page;
        ++(m_stats.m_u32Nodes);

        m_stats.m_nodesInLevel[n->m_level] = m_stats.m_nodesInLevel[n->m_level] + 1;
    }

    ++(m_stats.m_u64Writes);

    for (size_t cIndex = 0; cIndex < m_writeNodeCommands.size(); ++cIndex)
    {
        m_writeNodeCommands[cIndex]->execute(*n);
    }

    return page;
}

SpatialIndex::id_type SpatialIndex::RTree::RTree::writeNode_wo_modify_stats(Node* n)
{
    uint8_t* buffer;
    uint32_t dataLength;
    n->storeToByteArray(&buffer, dataLength);

    id_type page;
    if (n->m_identifier < 0) page = StorageManager::NewPage;
    else page = n->m_identifier;

    try
    {
        m_pStorageManager->storeByteArray(page, dataLength, buffer);
        delete[] buffer;
    }
    catch (InvalidPageException& e)
    {
        delete[] buffer;
        std::cerr << e.what() << std::endl;
        throw;
    }

    assert(n->m_identifier < 0);
    if (n->m_identifier < 0)
    {
        n->m_identifier = page;
        // ++(m_stats.m_u32Nodes);

        // m_stats.m_nodesInLevel[n->m_level] = m_stats.m_nodesInLevel[n->m_level] + 1;
    }

    // ++(m_stats.m_u64Writes);

    for (size_t cIndex = 0; cIndex < m_writeNodeCommands.size(); ++cIndex)
    {
        m_writeNodeCommands[cIndex]->execute(*n);
    }

    return page;
}

SpatialIndex::RTree::NodePtr SpatialIndex::RTree::RTree::readNode(id_type page)
{
    uint32_t dataLength;
    uint8_t* buffer;

    try
    {
        m_pStorageManager->loadByteArray(page, dataLength, &buffer);
    }
    catch (InvalidPageException& e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }

    try
    {
        uint32_t nodeType;
        memcpy(&nodeType, buffer, sizeof(uint32_t));

        NodePtr n;

        if (nodeType == PersistentIndex) n = m_indexPool.acquire();
        else if (nodeType == PersistentLeaf) n = m_leafPool.acquire();
        else throw Tools::IllegalStateException("readNode: failed reading the correct node type information");

        if (n.get() == nullptr)
        {
            if (nodeType == PersistentIndex) n = NodePtr(new Index(this, -1, 0), &m_indexPool);
            else if (nodeType == PersistentLeaf) n = NodePtr(new Leaf(this, -1), &m_leafPool);
        }

        //n->m_pTree = this;
        n->m_identifier = page;
        n->loadFromByteArray(buffer);

        ++(m_stats.m_u64Reads);

        for (size_t cIndex = 0; cIndex < m_readNodeCommands.size(); ++cIndex)
        {
            m_readNodeCommands[cIndex]->execute(*n);
        }

        delete[] buffer;
        return n;
    }
    catch (...)
    {
        delete[] buffer;
        throw;
    }
}

SpatialIndex::RTree::NodePtr SpatialIndex::RTree::RTree::readNode_wo_modify_stats(id_type page)
{
    uint32_t dataLength;
    uint8_t* buffer;

    try
    {
        m_pStorageManager->loadByteArray(page, dataLength, &buffer);
    }
    catch (InvalidPageException& e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }

    try
    {
        uint32_t nodeType;
        memcpy(&nodeType, buffer, sizeof(uint32_t));

        NodePtr n;

        if (nodeType == PersistentIndex) n = m_indexPool.acquire();
        else if (nodeType == PersistentLeaf) n = m_leafPool.acquire();
        else throw Tools::IllegalStateException("readNode: failed reading the correct node type information");

        if (n.get() == nullptr)
        {
            if (nodeType == PersistentIndex) n = NodePtr(new Index(this, -1, 0), &m_indexPool);
            else if (nodeType == PersistentLeaf) n = NodePtr(new Leaf(this, -1), &m_leafPool);
        }

        //n->m_pTree = this;
        n->m_identifier = page;
        n->loadFromByteArray(buffer);

        // ++(m_stats.m_u64Reads);

        for (size_t cIndex = 0; cIndex < m_readNodeCommands.size(); ++cIndex)
        {
            m_readNodeCommands[cIndex]->execute(*n);
        }

        delete[] buffer;
        return n;
    }
    catch (...)
    {
        delete[] buffer;
        throw;
    }
}

void SpatialIndex::RTree::RTree::deleteNode(Node* n)
{
    try
    {
        m_pStorageManager->deleteByteArray(n->m_identifier);
    }
    catch (InvalidPageException& e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }

    --(m_stats.m_u32Nodes);
    m_stats.m_nodesInLevel[n->m_level] = m_stats.m_nodesInLevel[n->m_level] - 1;

    for (size_t cIndex = 0; cIndex < m_deleteNodeCommands.size(); ++cIndex)
    {
        m_deleteNodeCommands[cIndex]->execute(*n);
    }
}

void SpatialIndex::RTree::RTree::rangeQuery(RangeQueryType type, const IShape& query, IVisitor& v)
{
    std::stack<NodePtr> st;
    NodePtr root = readNode(m_rootID);

    if (root->m_children > 0 && query.intersectsShape(root->m_nodeMBR)) st.push(root);

    while (!st.empty())
    {
        NodePtr n = st.top();
        st.pop();

        if (n->m_level == 0)
        {
            v.visitNode(*n);

            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                bool b;
                if (type == ContainmentQuery) b = query.containsShape(*(n->m_ptrMBR[cChild]));
                else b = query.intersectsShape(*(n->m_ptrMBR[cChild]));

                if (b)
                {
                    Data data = Data(n->m_pDataLength[cChild], n->m_pData[cChild], *(n->m_ptrMBR[cChild]),
                                     n->m_pIdentifier[cChild]);
                    v.visitData(data);
                    ++(m_stats.m_u64QueryResults);
                }
            }
        }
        else
        {
            v.visitNode(*n);

            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                if (query.intersectsShape(*(n->m_ptrMBR[cChild]))) st.push(readNode(n->m_pIdentifier[cChild]));
            }
        }
    }
}

uint64_t SpatialIndex::RTree::RTree::rangeQueryCount(RangeQueryType type, const IShape& query)
{
    uint64_t count = 0;
    std::stack<NodePtr> st;
    NodePtr root = readNode(m_rootID);

    if (root->m_children > 0 && query.intersectsShape(root->m_nodeMBR)) st.push(root);

    while (!st.empty())
    {
        NodePtr n = st.top();
        st.pop();

        if (n->m_level == 0)
        {
            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                bool b;
                if (type == ContainmentQuery) b = query.containsShape(*(n->m_ptrMBR[cChild]));
                else b = query.intersectsShape(*(n->m_ptrMBR[cChild]));

                if (b)
                {
                    Data data = Data(n->m_pDataLength[cChild], n->m_pData[cChild], *(n->m_ptrMBR[cChild]),
                                     n->m_pIdentifier[cChild]);
                    ++count;
                    ++(m_stats.m_u64QueryResults);
                }
            }
        }
        else
        {
            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                if (query.intersectsShape(*(n->m_ptrMBR[cChild]))) st.push(readNode(n->m_pIdentifier[cChild]));
            }
        }
    }
    return count;
}

uint64_t SpatialIndex::RTree::RTree::rangeQueryIOCost(RangeQueryType type, const IShape& query)
{
    // uint64_t count = 0;
    std::stack<NodePtr> st;
    NodePtr root = readNode(m_rootID);

    if (root->m_children > 0 && query.intersectsShape(root->m_nodeMBR)) st.push(root);

    uint64_t curr_num_io = m_stats.m_u64Reads;

    while (!st.empty())
    {
        NodePtr n = st.top();
        st.pop();

        if (n->m_level == 0)
        {
            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                bool b;
                if (type == ContainmentQuery) b = query.containsShape(*(n->m_ptrMBR[cChild]));
                else b = query.intersectsShape(*(n->m_ptrMBR[cChild]));

                if (b)
                {
                    Data data = Data(n->m_pDataLength[cChild], n->m_pData[cChild], *(n->m_ptrMBR[cChild]),
                                     n->m_pIdentifier[cChild]);
                    ++(m_stats.m_u64QueryResults);
                }
            }
        }
        else
        {
            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                if (query.intersectsShape(*(n->m_ptrMBR[cChild])))
                {
                    st.push(readNode(n->m_pIdentifier[cChild]));
                }
            }
        }
    }
    assert(curr_num_io < m_stats.m_u64Reads);
    return m_stats.m_u64Reads - curr_num_io;
}

void SpatialIndex::RTree::RTree::getOverlapMBRs(const IShape& query, std::vector<Region>& mbrs)
{
    std::stack<NodePtr> st;
    NodePtr root = readNode(m_rootID);

    if (root->m_children > 0 && query.intersectsShape(root->m_nodeMBR)) st.push(root);

    while (!st.empty())
    {
        NodePtr n = st.top();
        st.pop();

        if (n->m_level == 0)
        {
            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                bool b = query.intersectsShape(*(n->m_ptrMBR[cChild]));

                if (b)
                {
                    mbrs.push_back(*(n->m_ptrMBR[cChild]));
                    ++(m_stats.m_u64QueryResults);
                }
            }
        }
        else
        {
            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                if (query.intersectsShape(*(n->m_ptrMBR[cChild])))
                {
                    st.push(readNode(n->m_pIdentifier[cChild]));
                }
            }
        }
    }
}

void SpatialIndex::RTree::RTree::getRestrictedOverlapMBRIDs(const Region& query, const Region &ignore, std::unordered_set<id_type>& mbr_ids)
{
    std::stack<NodePtr> st;
    NodePtr root = readNode(m_rootID);

    if (root->m_children > 0 && query.intersectsShape(root->m_nodeMBR)) st.push(root);

    while (!st.empty())
    {
        NodePtr n = st.top();
        st.pop();

        if (n->m_level == 0)
        {
            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                bool b = query.intersectsShape(*(n->m_ptrMBR[cChild]));

                if (b)
                {
                    b = ignore.intersectsShape(*(n->m_ptrMBR[cChild]));
                    if (!b)
                    {
                        id_type id = *(reinterpret_cast<id_type*>(n->m_pData[cChild]));
                        mbr_ids.insert(id);
                        ++(m_stats.m_u64QueryResults);
                    }
                }
            }
        }
        else
        {
            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                if (query.intersectsShape(*(n->m_ptrMBR[cChild])))
                {
                    st.push(readNode(n->m_pIdentifier[cChild]));
                }
            }
        }
    }
}

void SpatialIndex::RTree::RTree::getRestrictedOverlapNodeIDs(const Region& query, const Region &ignore, std::unordered_set<id_type>& node_ids)
{
    std::stack<NodePtr> st;
    NodePtr root = readNode(m_rootID);

    if (root->m_children > 0 && query.intersectsShape(root->m_nodeMBR)) st.push(root);

    while (!st.empty())
    {
        NodePtr n = st.top();
        st.pop();

        if (!(ignore.intersectsShape(n->m_nodeMBR)))
        {
            node_ids.insert(n->m_identifier);
        }

        if (n->m_level > 0)
        {
            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                if (query.intersectsShape(*(n->m_ptrMBR[cChild])))
                {
                    st.push(readNode(n->m_pIdentifier[cChild]));
                }
            }
        }
    }
}


uint64_t SpatialIndex::RTree::RTree::countRestrictedOverlapMBRs(const Region& query, const Region &ignore)
{
    uint64_t result = 0;
    std::stack<NodePtr> st;
    NodePtr root = readNode(m_rootID);

    if (root->m_children > 0 && query.intersectsShape(root->m_nodeMBR)) st.push(root);

    while (!st.empty())
    {
        NodePtr n = st.top();
        st.pop();

        if (n->m_level == 0)
        {
            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                bool b = query.intersectsShape(*(n->m_ptrMBR[cChild]));

                if (b)
                {
                    b = ignore.intersectsShape(*(n->m_ptrMBR[cChild]));
                    if (!b)
                    {
                        id_type id = *(reinterpret_cast<id_type*>(n->m_pData[cChild]));
                        // mbr_ids.insert(id);
                        ++result;
                        ++(m_stats.m_u64QueryResults);
                    }
                }
            }
        }
        else
        {
            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                if (query.intersectsShape(*(n->m_ptrMBR[cChild])))
                {
                    st.push(readNode(n->m_pIdentifier[cChild]));
                }
            }
        }
    }
    return result;
}


void SpatialIndex::RTree::RTree::selfJoinQuery(id_type id1, id_type id2, const Region& r, IVisitor& vis)
{
    NodePtr n1 = readNode(id1);
    NodePtr n2 = readNode(id2);
    vis.visitNode(*n1);
    vis.visitNode(*n2);

    for (uint32_t cChild1 = 0; cChild1 < n1->m_children; ++cChild1)
    {
        if (r.intersectsRegion(*(n1->m_ptrMBR[cChild1])))
        {
            for (uint32_t cChild2 = 0; cChild2 < n2->m_children; ++cChild2)
            {
                if (
                    r.intersectsRegion(*(n2->m_ptrMBR[cChild2])) &&
                    n1->m_ptrMBR[cChild1]->intersectsRegion(*(n2->m_ptrMBR[cChild2])))
                {
                    if (n1->m_level == 0)
                    {
                        if (n1->m_pIdentifier[cChild1] != n2->m_pIdentifier[cChild2])
                        {
                            assert(n2->m_level == 0);

                            std::vector<const IData*> v;
                            Data e1(n1->m_pDataLength[cChild1], n1->m_pData[cChild1], *(n1->m_ptrMBR[cChild1]),
                                    n1->m_pIdentifier[cChild1]);
                            Data e2(n2->m_pDataLength[cChild2], n2->m_pData[cChild2], *(n2->m_ptrMBR[cChild2]),
                                    n2->m_pIdentifier[cChild2]);
                            v.push_back(&e1);
                            v.push_back(&e2);
                            vis.visitData(v);
                        }
                    }
                    else
                    {
                        Region rr = r.getIntersectingRegion(
                            n1->m_ptrMBR[cChild1]->getIntersectingRegion(*(n2->m_ptrMBR[cChild2])));
                        selfJoinQuery(n1->m_pIdentifier[cChild1], n2->m_pIdentifier[cChild2], rr, vis);
                    }
                }
            }
        }
    }
}


void SpatialIndex::RTree::RTree::visitSubTree(NodePtr subTree, IVisitor& v)
{
    std::stack<NodePtr> st;
    st.push(subTree);

    while (!st.empty())
    {
        NodePtr n = st.top();
        st.pop();
        v.visitNode(*n);

        if (n->m_level == 0)
        {
            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                Data data = Data(n->m_pDataLength[cChild], n->m_pData[cChild], *(n->m_ptrMBR[cChild]),
                                 n->m_pIdentifier[cChild]);
                v.visitData(data);
                ++(m_stats.m_u64QueryResults);
            }
        }
        else
        {
            for (uint32_t cChild = 0; cChild < n->m_children; ++cChild)
            {
                st.push(readNode(n->m_pIdentifier[cChild]));
            }
        }
    }
}

std::ostream& SpatialIndex::RTree::operator<<(std::ostream& os, const RTree& t)
{
    os << "Dimension: " << t.m_dimension << std::endl
        << "Fill factor: " << t.m_fillFactor << std::endl
        << "Index capacity: " << t.m_indexCapacity << std::endl
        << "Leaf capacity: " << t.m_leafCapacity << std::endl
        << "Tight MBRs: " << ((t.m_bTightMBRs) ? "enabled" : "disabled") << std::endl;

    if (t.m_treeVariant == RV_RSTAR
        || t.m_treeVariant == RV_BEST_SPLIT || t.m_treeVariant == RV_MODEL_SPLIT
        || t.m_treeVariant == RV_BEST_SUBTREE || t.m_treeVariant == RV_MODEL_SUBTREE
        || t.m_treeVariant == RV_BEST_BOTH || t.m_treeVariant == RV_MODEL_BOTH)
    {
        os << "Near minimum overlap factor: " << t.m_nearMinimumOverlapFactor << std::endl
            << "Reinsert factor: " << t.m_reinsertFactor << std::endl
            << "Split distribution factor: " << t.m_splitDistributionFactor << std::endl;
    }

    if (t.m_treeVariant == RV_BEST_SUBTREE)
    {
        os << "num_find_best_subtree = " << t.num_find_best_subtree << std::endl;
    }

    if (t.m_stats.getNumberOfNodesInLevel(0) > 0)
        os << "Utilization: " << 100 * t.m_stats.getNumberOfData() / (t.m_stats.getNumberOfNodesInLevel(0) * t.
                m_leafCapacity) << "%" << std::endl
            << t.m_stats;

    return os;
}


// --------------Customized functions------------------
void SpatialIndex::RTree::RTree::setDeviceConfigs()
{
    cpu_options = torch::TensorOptions().dtype(torch::kF32).device("cpu").requires_grad(false);//cfg.run.device
    if (torch::cuda::is_available())
    {
        device = "cuda:0";
        cuda_exists = true;
    }
    else
    {
        device = "cpu";
        cuda_exists = false;
    }
    std::cout << "device: " << device << std::endl;
    options = torch::TensorOptions().dtype(torch::kF32).device(device).requires_grad(false);//cfg.run.device
}

void SpatialIndex::RTree::get_split_rep_points(const Region &r, std::vector< std::vector<double> > &res)
{
    uint64_t dim = r.m_dimension;
    std::vector<double> low(r.m_pLow, r.m_pLow + dim);
    std::vector<double> high(r.m_pHigh, r.m_pHigh + dim);
    std::vector< std::vector<double> > vecs = {low, high };
    std::vector<double> middle(dim, 0);
    for (uint64_t d = 0; d < dim; ++d)
    {
        middle[d] = (low[d] + high[d]) / 2;
    }
    if (dim < 4)
    {
        vecs.emplace_back(middle);
    }

    uint64_t n_each_dim = vecs.size();
    uint64_t num_points = static_cast<uint64_t>(pow(n_each_dim, dim) + 0.5);
    for (uint64_t i = 0; i < num_points; ++i)
    {
        std::vector<double> point;
        uint64_t n = i;
        for (uint64_t d = 0; d < dim; ++d)
        {
            uint64_t index = n % n_each_dim;
            n /= n_each_dim;
            point.emplace_back(vecs[index][d]);
        }
        res.emplace_back(point);
    }
    if (dim >= 4)
    {
        res.emplace_back(middle);
    }
}

void SpatialIndex::RTree::RTree::initSplitRecording(const std::string &path)
{
    std::string parent_path = file_utils::parent_path(path);
    std::string filename = file_utils::file_name(path, false);
    filename += "_text.txt";
    std::string text_path = file_utils::path_join(parent_path, filename);

    if (append_splits)
    {
        split_records_file.open(path.c_str(), std::ios_base::out | std::ios_base::binary | std::ios_base::app);
        split_records_text_file.open(text_path.c_str(), std::ios_base::out | std::ios_base::app);
        if (!(file_utils::file_exists(path) || file_utils::file_exists(text_path)))
        {
            std::string err_msg = "In RTree::initSplitRecording, " + path + " or " + text_path + " does not exist.";
            throw Tools::BasicException(err_msg);
        }
    }
    else
    {
        split_records_file.open(path.c_str(), std::ios_base::out | std::ios_base::binary);
        split_records_text_file.open(text_path.c_str(), std::ios_base::out);

        int32_t dim = static_cast<int32_t>(m_dimension);
        if (m_leafCapacity != m_indexCapacity)
        {
            throw Tools::IsNotEqualException<uint32_t>("RTree::initSplitRecording", "m_leafCapacity", "m_indexCapacity", m_leafCapacity, m_indexCapacity);
        }
        int32_t num_mbrs = static_cast<int32_t>(m_leafCapacity + 1);
        int32_t num_geometric_properties = SplitInfo::get_num_geometric_properties();

        uint32_t nodeSPF = static_cast<uint32_t>(
            std::floor(num_mbrs * m_splitDistributionFactor));
        uint32_t splitDistribution = num_mbrs - (2 * nodeSPF) + 2;
        int32_t num_splits_threshold = m_dimension * 2 * splitDistribution;

        split_records_file.write(reinterpret_cast<char*>(&dim), sizeof(int32_t));
        split_records_file.write(reinterpret_cast<char*>(&num_mbrs), sizeof(int32_t));
        split_records_file.write(reinterpret_cast<char*>(&num_splits_threshold), sizeof(int32_t));
        split_records_file.write(reinterpret_cast<char*>(&num_geometric_properties), sizeof(int32_t));
        split_records_file.write(reinterpret_cast<char*>(lbds.data()), sizeof(double) * m_dimension);
        split_records_file.write(reinterpret_cast<char*>(ubds.data()), sizeof(double) * m_dimension);

        split_records_text_file << dim << "," << num_mbrs << "," << split_model_config_.max_num_splits << "," << num_geometric_properties;
        for (int j = 0; j < m_dimension; ++j)
        {
            split_records_text_file << "," << lbds[j];
        }

        for (int j = 0; j < m_dimension; ++j)
        {
            split_records_text_file << "," << ubds[j];
        }
        split_records_text_file << std::endl;
    }
}


void SpatialIndex::RTree::RTree::writeASplitRecord(
    Region& m_nodeMBR, std::vector<Region>& child_mbrs,
    std::vector<SplitInfo>& split_infos, std::vector<double> &split_scores,
    bool isLeaf)
{
    int32_t isLeafInt = static_cast<int32_t>(isLeaf);
    int32_t dim = static_cast<int32_t>(m_dimension);
    int32_t num_mbrs = static_cast<int32_t>(child_mbrs.size());
    int32_t num_splits = static_cast<int32_t>(split_infos.size()) - 1;
    std::vector< std::vector<double> > representative_points;
    get_split_rep_points(m_nodeMBR, representative_points);
    uint64_t num_representative_points = representative_points.size();
    // int32_t num_split_context_rep_points = static_cast<int32_t>(num_representative_points + num_mbrs);

    split_records_file.write(reinterpret_cast<char*>(&isLeafInt), sizeof(int32_t));
    split_records_file.write(reinterpret_cast<char*>(&num_splits), sizeof(int32_t));
    split_records_file.write(reinterpret_cast<char*>(m_nodeMBR.m_pLow), dim * sizeof(double));
    split_records_file.write(reinterpret_cast<char*>(m_nodeMBR.m_pHigh), dim * sizeof(double));

    for (std::vector<double> &point: representative_points)
    {
        split_records_file.write(reinterpret_cast<char*>(point.data()), dim * sizeof(double));
    }

    std::vector<double> point(dim, 0);
    for (Region &mbr: child_mbrs)
    {
        for (int32_t d = 0; d < dim; ++d)
        {
            point[d] = (mbr.m_pLow[d] + mbr.m_pHigh[d]) / 2;
        }
        split_records_file.write(reinterpret_cast<char*>(point.data()), dim * sizeof(double));
    }

    for (int32_t i = 0; i <= num_splits; ++i)
    {
        SplitInfo &split_info = split_infos[i];
        for (std::vector<double> &point: split_info.candi_rep_points)
        {
            split_records_file.write(reinterpret_cast<char*>(point.data()), dim * sizeof(double));
        }
    }

    for (int32_t i = 0; i <= num_splits; ++i)
    {
        SplitInfo &split_info = split_infos[i];
        split_records_file.write(reinterpret_cast<char*>(split_info.geometric_properties.data()), (split_info.geometric_properties).size() * sizeof(double));
    }

    for (int32_t i = 0; i <= num_splits; ++i)
    {
        split_records_file.write(reinterpret_cast<char*>(&(split_scores[i])), sizeof(double));
    }

    // -------------------------- text --------------------------
    split_records_text_file << (num_representative_points + child_mbrs.size()) << "," << isLeafInt << "," << num_splits << "||" << (m_nodeMBR.m_pLow)[0];;
    for (int j = 1; j < dim; ++j)
    {
        split_records_text_file << "," << (m_nodeMBR.m_pLow)[j];
    }
    split_records_text_file << "~" << (m_nodeMBR.m_pHigh)[0];
    for (int j = 1; j < dim; ++j)
    {
        split_records_text_file << "," << (m_nodeMBR.m_pHigh)[j];
    }

    split_records_text_file << "||";

    for (uint64_t i = 0; i < num_representative_points; ++i)
    {
        for (int j = 0; j < dim - 1; ++j)
        {
            split_records_text_file << representative_points[i][j] << ",";
        }
        split_records_text_file << representative_points[i][dim - 1] << "~";
    }
    for (Region &mbr: child_mbrs)
    {
        for (int32_t d = 0; d < dim; ++d)
        {
            point[d] = (mbr.m_pLow[d] + mbr.m_pHigh[d]) / 2;
        }
        for (int j = 0; j < dim - 1; ++j)
        {
            split_records_text_file << point[j] << ",";
        }
        split_records_text_file << point[dim - 1] << "~";
    }

    split_records_text_file << "||";
    for (int32_t i = 0; i <= num_splits; ++i)
    {
        SplitInfo &split_info = split_infos[i];
        split_records_text_file << split_info.splitAxis << ": ";

        for (std::vector<double> &point: split_info.candi_rep_points)
        {
            int j = 0;
            for (; j < dim-1; ++j)
            {
                split_records_text_file << point[j] << ",";
            }
            split_records_text_file << point[j] << "~";
        }

        split_records_text_file << "**";
    }

    split_records_text_file << "||";
    for (int32_t i = 0; i <= num_splits; ++i)
    {
        std::vector<double> &geometric_properties = split_infos[i].geometric_properties;
        split_records_text_file << geometric_properties[0];
        for (int j = 1; j < geometric_properties.size(); ++j)
        {
            split_records_text_file << "," << geometric_properties[j] ;
        }
        split_records_text_file << "**";
    }

    split_records_text_file << "||" << split_scores[0];

    for (int i = 1; i <= num_splits; ++i)
    {
        split_records_text_file << "," << split_scores[i];
    }

    split_records_text_file << std::endl;
}

void SpatialIndex::RTree::RTree::loadSplitNormalizationInfos()
{
    // std::cout << "split_normalization_infos_path = " << split_normalization_infos_path << std::endl;
    std::ifstream fin(split_normalization_infos_path.c_str(), std::ios_base::in | std::ios_base::binary);

    fin.read(reinterpret_cast<char*>(&split_model_config_.max_num_splits), sizeof(int64_t));
    fin.read(reinterpret_cast<char*>(&split_model_config_.num_split_context_rep_points), sizeof(int64_t));
    fin.read(reinterpret_cast<char*>(&split_model_config_.num_candi_rep_points_per_split), sizeof(int64_t));
    fin.read(reinterpret_cast<char*>(&split_model_config_.num_maps_each_point), sizeof(int64_t));
    fin.read(reinterpret_cast<char*>(&split_model_config_.num_geometric_properties), sizeof(int64_t));
    split_model_config_.points_mean.resize(split_model_config_.num_maps_each_point * m_dimension);
    split_model_config_.points_std.resize(split_model_config_.num_maps_each_point * m_dimension);
    split_model_config_.geometric_properties_mean.resize(split_model_config_.num_geometric_properties);
    split_model_config_.geometric_properties_std.resize(split_model_config_.num_geometric_properties);
    fin.read(reinterpret_cast<char*>(split_model_config_.points_mean.data()), sizeof(double) * split_model_config_.num_maps_each_point * m_dimension);
    fin.read(reinterpret_cast<char*>(split_model_config_.points_std.data()), sizeof(double) * split_model_config_.num_maps_each_point * m_dimension);
    fin.read(reinterpret_cast<char*>(split_model_config_.geometric_properties_mean.data()), sizeof(double) * split_model_config_.num_geometric_properties);
    fin.read(reinterpret_cast<char*>(split_model_config_.geometric_properties_std.data()), sizeof(double) * split_model_config_.num_geometric_properties);
    fin.close();

    std::vector<int64_t> _split_context_rep_points_shape = {1, split_model_config_.num_split_context_rep_points, split_model_config_.num_maps_each_point * m_dimension};
    std::vector<int64_t> _split_candi_rep_points_shape = {1, split_model_config_.max_num_splits * split_model_config_.num_candi_rep_points_per_split, split_model_config_.num_maps_each_point * m_dimension};
    std::vector<int64_t> _split_geometric_properties_shape = {1, split_model_config_.max_num_splits, split_model_config_.num_geometric_properties};
    split_model_config_.split_context_rep_points_shape = _split_context_rep_points_shape;
    split_model_config_.split_candi_rep_points_shape = _split_candi_rep_points_shape;
    split_model_config_.split_geometric_properties_shape = _split_geometric_properties_shape;

    // std::cout << "max_num_splits = " << split_model_config_.max_num_splits << std::endl <<
    //     "num_split_context_rep_points = " << split_model_config_.num_split_context_rep_points << std::endl <<
    //         "num_candi_rep_points_per_split = " << split_model_config_.num_candi_rep_points_per_split << std::endl <<
    //             "num_maps_each_point = " << split_model_config_.num_maps_each_point << std::endl <<
    //                 "num_geometric_properties = " << split_model_config_.num_geometric_properties << std::endl;
    // print_utils::print_vec(split_model_config_.points_mean, "points_mean");
    // print_utils::print_vec(split_model_config_.points_std, "points_std");
    // print_utils::print_vec(split_model_config_.geometric_properties_mean, "geometric_properties_mean");
    // print_utils::print_vec(split_model_config_.geometric_properties_std, "geometric_properties_std");
}

void SpatialIndex::RTree::RTree::loadSplitModel()
{
    if (cuda_exists) split_model = torch::jit::load(split_model_path.c_str(), torch::kCUDA);
    else split_model = torch::jit::load(split_model_path.c_str());
    split_model.to(device);
    std::cout << "----------------The split model is successfully loaded.-------------------" << std::endl;
}

int64_t SpatialIndex::RTree::RTree::splitPredict(Region& m_nodeMBR, std::vector<Region>& child_mbrs, std::vector<SplitInfo>& split_infos, bool isLeaf)
{
    // int64_t isLeafInt = static_cast<int64_t>(isLeaf);
    int64_t dim = static_cast<int64_t>(m_dimension);
    int64_t num_splits = static_cast<int64_t>(split_infos.size()) - 1;
    std::vector< std::vector<double> > representative_points;
    get_split_rep_points(m_nodeMBR, representative_points);

    // std::cout << "isLeafInt = " << isLeafInt << ", dim = " << dim << ", num_mbrs = " << num_mbrs << ", num_splits = " << num_splits << std::endl;
    // std::cout << "max_num_splits = " << max_num_splits << ", dim = " << dim << ", num_mbrs = " << num_mbrs << ", num_geometric_properties = " << num_geometric_properties << std::endl;
    if (split_debug)
    {
        // split_debug = false;
        std::cout << "points_mean: " << split_model_config_.points_mean[0];
        for (uint64_t i = 1; i < m_dimension; ++i)
        {
            std::cout << ", " << split_model_config_.points_mean[i];
        }
        std::cout << std::endl;

        std::cout << "points_std: " << split_model_config_.points_std[0];
        for (uint64_t i = 1; i < m_dimension; ++i)
        {
            std::cout << ", " << split_model_config_.points_std[i];
        }
        std::cout << std::endl;

        std::cout << "geometric_properties_mean: " << split_model_config_.geometric_properties_mean[0];
        for (uint64_t i = 1; i < m_dimension; ++i)
        {
            std::cout << ", " << split_model_config_.geometric_properties_mean[i];
        }
        std::cout << std::endl;

        std::cout << "geometric_properties_std: " << split_model_config_.geometric_properties_std[0];
        for (uint64_t i = 1; i < m_dimension; ++i)
        {
            std::cout << ", " << split_model_config_.geometric_properties_std[i];
        }
        std::cout << std::endl;
    }

    // SplitInfo padding_split_info(std::move(split_infos.back()));
    // SplitInfo padding_split_info(split_infos.back());
    if (split_debug)
    {
        std::cout << "1. It is OK here." << std::endl;
        std::cout << "num_splits = " << num_splits << ", max_num_splits = " << split_model_config_.max_num_splits << std::endl;
    }

    if (num_splits > split_model_config_.max_num_splits)
    {
        split_infos.pop_back();
        std::sort(split_infos.begin(), split_infos.end(), SplitInfo::compare);
        split_infos.resize(split_model_config_.max_num_splits);
        num_splits = split_model_config_.max_num_splits;
    }
    if (split_debug)
    {
        std::cout << "2. It is OK here." << std::endl;
    }

    std::vector<double> lbds_local;
    std::vector<double> diff_local;

    for (int64_t d = 0; d < dim; ++d)
    {
        lbds_local.emplace_back(m_nodeMBR.m_pLow[d]);
        diff_local.emplace_back(m_nodeMBR.m_pHigh[d] - m_nodeMBR.m_pLow[d]);
    }
    if (split_debug)
    {
        std::cout << "3. It is OK here." << std::endl;
    }

    input_split_context_rep_points.clear();
    for (std::vector<double> &point: representative_points)
    {
        std::vector<float> expanded_point;
        for (int64_t d = 0; d < dim; ++d)
        {
            double x = point[d];
            double expanded_x_local = (x - lbds_local[d]) / diff_local[d];
            double expanded_x_global = (x - lbds[d]) / (ubds[d] - lbds[d]);

            expanded_x_local = (expanded_x_local - split_model_config_.points_mean[2 * d]) / split_model_config_.points_std[2 * d];
            expanded_x_global = (expanded_x_global - split_model_config_.points_mean[2 * d + 1]) / split_model_config_.points_std[2 * d + 1];

            input_split_context_rep_points.emplace_back(static_cast<float>(expanded_x_local));
            input_split_context_rep_points.emplace_back(static_cast<float>(expanded_x_global));
        }
    }
    if (split_debug)
    {
        std::cout << "4. It is OK here." << std::endl;
    }

    std::vector<double> point(dim, 0);
    for (Region &mbr: child_mbrs)
    {
        for (int64_t d = 0; d < dim; ++d)
        {
            double x = (mbr.m_pLow[d] + mbr.m_pHigh[d]) / 2;
            double expanded_x_local = (x - lbds_local[d]) / diff_local[d];
            double expanded_x_global = (x - lbds[d]) / (ubds[d] - lbds[d]);

            expanded_x_local = (expanded_x_local - split_model_config_.points_mean[2 * d]) / split_model_config_.points_std[2 * d];
            expanded_x_global = (expanded_x_global - split_model_config_.points_mean[2 * d + 1]) / split_model_config_.points_std[2 * d + 1];

            input_split_context_rep_points.emplace_back(static_cast<float>(expanded_x_local));
            input_split_context_rep_points.emplace_back(static_cast<float>(expanded_x_global));
        }
    }
    if (split_debug)
    {
        std::cout << "5. It is OK here." << std::endl;
    }

    input_split_candi_rep_points.clear();
    for (int64_t i = 0; i < num_splits; ++i)
    {
        SplitInfo &split_info = split_infos[i];
        for (std::vector<double> &_point: split_info.candi_rep_points)
        {
            for (int64_t d = 0; d < dim; ++d)
            {
                double x = _point[d];
                double expanded_x_local = (x - lbds_local[d]) / diff_local[d];
                double expanded_x_global = (x - lbds[d]) / (ubds[d] - lbds[d]);

                expanded_x_local = (expanded_x_local - split_model_config_.points_mean[2 * d]) / split_model_config_.points_std[2 * d];
                expanded_x_global = (expanded_x_global - split_model_config_.points_mean[2 * d + 1]) / split_model_config_.points_std[2 * d + 1];


                input_split_candi_rep_points.emplace_back(static_cast<float>(expanded_x_local));
                input_split_candi_rep_points.emplace_back(static_cast<float>(expanded_x_global));
            }
        }
    }
    if (split_debug)
    {
        std::cout << "6. It is OK here." << std::endl;
    }

    if (num_splits < split_model_config_.max_num_splits)
    {
        std::vector<float> padding_expanded_point;
        SplitInfo &split_info = split_infos[num_splits];
        for (std::vector<double> &_point: split_info.candi_rep_points)
        {
            for (int64_t d = 0; d < dim; ++d)
            {
                double x = _point[d];
                double expanded_x_local = (x - lbds_local[d]) / diff_local[d];
                double expanded_x_global = (x - lbds[d]) / (ubds[d] - lbds[d]);

                expanded_x_local = (expanded_x_local - split_model_config_.points_mean[2 * d]) / split_model_config_.points_std[2 * d];
                expanded_x_global = (expanded_x_global - split_model_config_.points_mean[2 * d + 1]) / split_model_config_.points_std[2 * d + 1];

                padding_expanded_point.emplace_back(static_cast<float>(expanded_x_local));
                padding_expanded_point.emplace_back(static_cast<float>(expanded_x_global));
            }
        }

        for (int64_t i = num_splits; i < split_model_config_.max_num_splits; ++i)
        {
            input_split_candi_rep_points.insert(input_split_candi_rep_points.end(), padding_expanded_point.begin(), padding_expanded_point.end());
        }
    }
    if (split_debug)
    {
        std::cout << "7. It is OK here." << std::endl;
    }

    input_split_geometric_properties.clear();
    for (int64_t i = 0; i < num_splits; ++i)
    {
        SplitInfo &split_info = split_infos[i];
        for (int64_t j = 0; j < split_model_config_.num_geometric_properties; ++j)
        {
            double x = (split_info.geometric_properties[j] - split_model_config_.geometric_properties_mean[j]) / split_model_config_.geometric_properties_std[j];
            input_split_geometric_properties.emplace_back(static_cast<float>(x));
        }
    }
    if (split_debug)
    {
        std::cout << "8. It is OK here." << std::endl;
    }
    if (num_splits < split_model_config_.max_num_splits)
    {
        std::vector<float> padding_geometric_properties;
        SplitInfo &split_info = split_infos[num_splits];
        for (int64_t j = 0; j < split_model_config_.num_geometric_properties; ++j)
        {
            double x = (split_info.geometric_properties[j] - split_model_config_.geometric_properties_mean[j]) / split_model_config_.geometric_properties_std[j];
            padding_geometric_properties.emplace_back(static_cast<float>(x));
        }
        for (int64_t i = num_splits; i < split_model_config_.max_num_splits; ++i)
        {
            input_split_geometric_properties.insert(input_split_geometric_properties.end(), padding_geometric_properties.begin(), padding_geometric_properties.end());
        }
    }


    if (split_debug)
    {
        std::cout << "9. It is OK here." << std::endl;
        // split_debug = false;
        // std::cout << "representative_points: " << input_split_context_rep_points[0];
        // for (uint64_t i = 1; i < input_split_context_rep_points.size(); ++i)
        // {
        //     std::cout << ", " << input_split_context_rep_points[i];
        // }
        // std::cout << std::endl;
        //
        // std::cout << "split_candi_rep_points: " << input_split_candi_rep_points[0];
        // for (uint64_t i = 1; i < input_split_candi_rep_points.size(); ++i)
        // {
        //     std::cout << ", " << input_split_candi_rep_points[i];
        // }
        // std::cout << std::endl;
        //
        // std::cout << "split_geometric_properties: " << input_split_geometric_properties[0];
        // for (uint64_t i = 1; i < input_split_geometric_properties.size(); ++i)
        // {
        //     std::cout << ", " << input_split_geometric_properties[i];
        // }
        // std::cout << std::endl;

        std::cout << "input_split_context_rep_points.size = " << input_split_context_rep_points.size() << ". It should be " << split_model_config_.split_context_rep_points_shape[1] * split_model_config_.split_context_rep_points_shape[2] << std::endl;
        std::cout << "input_split_candi_rep_points.size = " << input_split_candi_rep_points.size() << ". It should be " << split_model_config_.split_candi_rep_points_shape[1] * split_model_config_.split_candi_rep_points_shape[2] << std::endl;
        std::cout << "input_split_geometric_properties.size = " << input_split_geometric_properties.size() << ". It should be " << split_model_config_.split_geometric_properties_shape[1] * split_model_config_.split_geometric_properties_shape[2] << std::endl;
    }


    // tensor_split_context_rep_points = torch::from_blob(input_split_context_rep_points.data(), split_context_rep_points_shape, cpu_options);
    // tensor_split_candi_rep_points = torch::from_blob(input_split_candi_rep_points.data(), split_candi_rep_points_shape, cpu_options);
    // tensor_split_geometric_properties = torch::from_blob(input_split_geometric_properties.data(), split_geometric_properties_shape, cpu_options);

    torch::Tensor tensor_split_context_rep_points = torch::from_blob(input_split_context_rep_points.data(), split_model_config_.split_context_rep_points_shape, cpu_options);
    torch::Tensor tensor_split_candi_rep_points = torch::from_blob(input_split_candi_rep_points.data(), split_model_config_.split_candi_rep_points_shape, cpu_options);
    torch::Tensor tensor_split_geometric_properties = torch::from_blob(input_split_geometric_properties.data(), split_model_config_.split_geometric_properties_shape, cpu_options);
    // std::cout << "11. It is OK here." << std::endl;
    torch::Tensor pred_scores_tensor = split_model.forward({tensor_split_context_rep_points.to(device), tensor_split_candi_rep_points.to(device), tensor_split_geometric_properties.to(device)}).toTensor();

    std::vector<float> pred_scores;
    for (int64_t j = 0; j < num_splits; ++j)
    {
        pred_scores.push_back(pred_scores_tensor.index({0, j}).item<float>());
    }
    int64_t pred_index = vector_utils::argmax(pred_scores);
    if (split_debug)
    {
        std::cout << "9. It is OK here." << std::endl;
        std::cout << "pred_index = " << pred_index << ", num_splits = " << num_splits << ", max_num_splits = " << split_model_config_.max_num_splits << std::endl;
    }

    return pred_index;
}


void SpatialIndex::RTree::RTree::initSubtreeRecording(const std::string &path)
{
    std::string parent_path = file_utils::parent_path(path);
    std::string filename = file_utils::file_name(path, false);
    filename += "_text.txt";
    std::string text_path = file_utils::path_join(parent_path, filename);

    if (append_subtrees)
    {
        subtree_records_file.open(path.c_str(), std::ios_base::out | std::ios_base::binary | std::ios_base::app);
        subtree_records_text_file.open(text_path.c_str(), std::ios_base::out | std::ios_base::app);
        if (!(file_utils::file_exists(path) || file_utils::file_exists(text_path)))
        {
            std::string err_msg = "In RTree::initSubtreeRecording, " + path + " or " + text_path + " does not exist.";
            throw Tools::BasicException(err_msg);
        }
    }
    else
    {
        subtree_records_file.open(path.c_str(), std::ios_base::out | std::ios_base::binary);
        subtree_records_text_file.open(text_path.c_str(), std::ios_base::out);

        int32_t dim = static_cast<int32_t>(m_dimension);
        if (m_leafCapacity != m_indexCapacity)
        {
            throw Tools::IsNotEqualException<uint32_t>("RTree::initSubtreeRecording", "m_leafCapacity", "m_indexCapacity", m_leafCapacity, m_indexCapacity);
        }
        int32_t num_subtrees_threshold = static_cast<int32_t>(m_leafCapacity + 1);
        int32_t num_geometric_properties = SubtreeInfo::get_num_geometric_properties();

        subtree_records_file.write(reinterpret_cast<char*>(&dim), sizeof(int32_t));
        subtree_records_file.write(reinterpret_cast<char*>(&num_subtrees_threshold), sizeof(int32_t));
        subtree_records_file.write(reinterpret_cast<char*>(&num_geometric_properties), sizeof(int32_t));
        subtree_records_file.write(reinterpret_cast<char*>(lbds.data()), sizeof(double) * m_dimension);
        subtree_records_file.write(reinterpret_cast<char*>(ubds.data()), sizeof(double) * m_dimension);

        subtree_records_text_file << dim << "," << num_subtrees_threshold << "," << num_geometric_properties;
        for (int j = 0; j < m_dimension; ++j)
        {
            subtree_records_text_file << "," << lbds[j];
        }

        for (int j = 0; j < m_dimension; ++j)
        {
            subtree_records_text_file << "," << ubds[j];
        }
        subtree_records_text_file << std::endl;
    }
}

void SpatialIndex::RTree::get_region_boundary_points(const Region &r, std::vector< std::vector<double> > &res)
{
    uint64_t dim = r.m_dimension;
    std::vector<double> low(r.m_pLow, r.m_pLow + dim);
    std::vector<double> high(r.m_pHigh, r.m_pHigh + dim);
    std::vector< std::vector<double> > vecs = {low, high };

    const uint64_t n_each_dim = 2;
    uint64_t num_points = static_cast<uint64_t>(pow(n_each_dim, dim) + 0.5);
    for (uint64_t i = 0; i < num_points; ++i)
    {
        std::vector<double> point;
        uint64_t n = i;
        for (uint64_t d = 0; d < dim; ++d)
        {
            uint64_t index = n % n_each_dim;
            n /= n_each_dim;
            point.emplace_back(vecs[index][d]);
        }
        res.emplace_back(point);
    }
}

void SpatialIndex::RTree::RTree::writeASubtreeRecord(const Region &object, const Region &m_nodeMBR, std::vector<SubtreeInfo>& subtree_infos, std::vector<double> &subtree_scores, uint32_t level)
{
    int32_t dim = static_cast<int32_t>(m_dimension);
    int32_t num_subtrees = static_cast<int32_t>(subtree_infos.size()) - 1;
    std::vector< std::vector<double> > subtree_context_rep_points;
    get_region_boundary_points(object, subtree_context_rep_points);
    int32_t num_subtree_context_rep_points = static_cast<int32_t>(subtree_context_rep_points.size());

    // subtree_records_file.write(reinterpret_cast<char*>(&isLeafInt), sizeof(int32_t));
    // subtree_records_file.write(reinterpret_cast<char*>(&num_subtree_context_rep_points), sizeof(int32_t));
    subtree_records_file.write(reinterpret_cast<char*>(&num_subtrees), sizeof(int32_t));
    subtree_records_file.write(reinterpret_cast<char*>(m_nodeMBR.m_pLow), dim * sizeof(double));
    subtree_records_file.write(reinterpret_cast<char*>(m_nodeMBR.m_pHigh), dim * sizeof(double));

    for (std::vector<double> &point: subtree_context_rep_points)
    {
        subtree_records_file.write(reinterpret_cast<char*>(point.data()), dim * sizeof(double));
    }

    // -------------------------- text --------------------------
    subtree_records_text_file << num_subtree_context_rep_points << "," << num_subtrees << "||" << (m_nodeMBR.m_pLow)[0];;
    for (int j = 1; j < dim; ++j)
    {
        subtree_records_text_file << "," << (m_nodeMBR.m_pLow)[j];
    }
    subtree_records_text_file << "~" << (m_nodeMBR.m_pHigh)[0];
    for (int j = 1; j < dim; ++j)
    {
        subtree_records_text_file << "," << (m_nodeMBR.m_pHigh)[j];
    }

    subtree_records_text_file << "||";

    for (uint64_t i = 0; i < num_subtree_context_rep_points; ++i)
    {
        for (int j = 0; j < dim - 1; ++j)
        {
            subtree_records_text_file << subtree_context_rep_points[i][j] << ",";
        }
        subtree_records_text_file << subtree_context_rep_points[i][dim - 1] << "~";
    }
    subtree_records_text_file << "||";
    // ----------------------------------------------------

    for (int32_t i = 0; i <= num_subtrees; ++i)
    {
        SubtreeInfo &subtree_info = subtree_infos[i];
        std::vector< std::vector<double> > _points;
        get_region_boundary_points(subtree_info.bb, _points);
        for (std::vector<double> &_point: _points)
        {
            subtree_records_file.write(reinterpret_cast<char*>(_point.data()), dim * sizeof(double));
            for (int j = 0; j < dim - 1; ++j)
            {
                subtree_records_text_file << _point[j] << ",";
            }
            subtree_records_text_file << _point[dim - 1] << "~";
        }
        // for (std::vector<double> &_point: split_info.candi_rep_points)
        // {
        //     subtree_records_file.write(reinterpret_cast<char*>(point.data()), dim * sizeof(double));
        // }
    }

    for (int32_t i = 0; i <= num_subtrees; ++i)
    {
        SubtreeInfo &subtree_info = subtree_infos[i];
        subtree_records_file.write(reinterpret_cast<char*>(subtree_info.geometric_properties.data()), (subtree_info.geometric_properties).size() * sizeof(double));
    }

    for (int32_t i = 0; i <= num_subtrees; ++i)
    {
        subtree_records_file.write(reinterpret_cast<char*>(&(subtree_scores[i])), sizeof(double));
    }

    // -------------------------- text --------------------------

    subtree_records_text_file << "||";
    for (int32_t i = 0; i <= num_subtrees; ++i)
    {
        std::vector<double> &geometric_properties = subtree_infos[i].geometric_properties;
        subtree_records_text_file << geometric_properties[0];
        for (int j = 1; j < geometric_properties.size(); ++j)
        {
            subtree_records_text_file << "," << geometric_properties[j] ;
        }
        subtree_records_text_file << "**";
    }
    subtree_records_text_file << "||" << subtree_scores[0];

    for (int i = 1; i <= num_subtrees; ++i)
    {
        subtree_records_text_file << "," << subtree_scores[i];
    }

    subtree_records_text_file << std::endl;
}

void SpatialIndex::RTree::RTree::loadSubtreeNormalizationInfos()
{
    std::cout << "subtree_normalization_infos_path = " << subtree_normalization_infos_path << std::endl;
    std::ifstream fin(subtree_normalization_infos_path.c_str(), std::ios_base::in | std::ios_base::binary);

    fin.read(reinterpret_cast<char*>(&subtree_model_config_.max_num_subtrees), sizeof(int64_t));
    fin.read(reinterpret_cast<char*>(&subtree_model_config_.num_subtree_context_rep_points), sizeof(int64_t));
    fin.read(reinterpret_cast<char*>(&subtree_model_config_.num_candi_rep_points_per_subtree), sizeof(int64_t));
    fin.read(reinterpret_cast<char*>(&subtree_model_config_.num_maps_each_point), sizeof(int64_t));
    fin.read(reinterpret_cast<char*>(&subtree_model_config_.num_geometric_properties), sizeof(int64_t));
    std::cout << "max_num_subtrees = " << subtree_model_config_.max_num_subtrees << std::endl <<
        "num_subtree_context_rep_points = " << subtree_model_config_.num_subtree_context_rep_points << std::endl <<
            "num_candi_rep_points_per_subtree = " << subtree_model_config_.num_candi_rep_points_per_subtree << std::endl <<
                "num_maps_each_point = " << subtree_model_config_.num_maps_each_point << std::endl <<
                    "num_geometric_properties = " << subtree_model_config_.num_geometric_properties << std::endl;
    // int64_t buffer_size = (num_maps_each_point * m_dimension + num_geometric_properties) * 2;
    // std::vector<double> buffer(buffer_size, 0);
    subtree_model_config_.points_mean.resize(subtree_model_config_.num_maps_each_point * m_dimension);
    subtree_model_config_.points_std.resize(subtree_model_config_.num_maps_each_point * m_dimension);
    subtree_model_config_.geometric_properties_mean.resize(subtree_model_config_.num_geometric_properties);
    subtree_model_config_.geometric_properties_std.resize(subtree_model_config_.num_geometric_properties);
    fin.read(reinterpret_cast<char*>(subtree_model_config_.points_mean.data()), sizeof(double) * subtree_model_config_.num_maps_each_point * m_dimension);
    fin.read(reinterpret_cast<char*>(subtree_model_config_.points_std.data()), sizeof(double) * subtree_model_config_.num_maps_each_point * m_dimension);
    fin.read(reinterpret_cast<char*>(subtree_model_config_.geometric_properties_mean.data()), sizeof(double) * subtree_model_config_.num_geometric_properties);
    fin.read(reinterpret_cast<char*>(subtree_model_config_.geometric_properties_std.data()), sizeof(double) * subtree_model_config_.num_geometric_properties);
    fin.close();

    std::vector<int64_t> _subtree_context_rep_points_shape = {1, subtree_model_config_.num_subtree_context_rep_points, subtree_model_config_.num_maps_each_point * m_dimension};
    std::vector<int64_t> _subtree_candi_rep_points_shape = {1, subtree_model_config_.max_num_subtrees * subtree_model_config_.num_candi_rep_points_per_subtree, subtree_model_config_.num_maps_each_point * m_dimension};
    std::vector<int64_t> _subtree_geometric_properties_shape = {1, subtree_model_config_.max_num_subtrees, subtree_model_config_.num_geometric_properties};
    subtree_model_config_.subtree_context_rep_points_shape = _subtree_context_rep_points_shape;
    subtree_model_config_.subtree_candi_rep_points_shape = _subtree_candi_rep_points_shape;
    subtree_model_config_.subtree_geometric_properties_shape = _subtree_geometric_properties_shape;


    print_utils::print_vec(subtree_model_config_.points_mean, "points_mean");
    print_utils::print_vec(subtree_model_config_.points_std, "points_std");
    print_utils::print_vec(subtree_model_config_.geometric_properties_mean, "geometric_properties_mean");
    print_utils::print_vec(subtree_model_config_.geometric_properties_std, "geometric_properties_std");
}

void SpatialIndex::RTree::RTree::loadSubtreeModel()
{
    if (cuda_exists) subtree_model = torch::jit::load(subtree_model_path.c_str(), torch::kCUDA);
    else subtree_model = torch::jit::load(subtree_model_path.c_str());
    subtree_model.to(device);
    std::cout << "----------------The subtree model is successfully loaded.-------------------" << std::endl;
}

int64_t SpatialIndex::RTree::RTree::subtreePredict(const Region &object, const Region& m_nodeMBR, std::vector<SubtreeInfo>& subtree_infos,  uint32_t level)
{
    int32_t dim = static_cast<int32_t>(m_dimension);
    int32_t num_subtrees = static_cast<int32_t>(subtree_infos.size()) - 1;
    std::vector< std::vector<double> > subtree_context_rep_points;
    get_region_boundary_points(object, subtree_context_rep_points);
    // int32_t num_subtree_context_rep_points = static_cast<int32_t>(subtree_context_rep_points.size());

    if (subtree_debug)
    {
        // subtree_debug = false;
        std::cout << "points_mean: " << subtree_model_config_.points_mean[0];
        for (uint64_t i = 1; i < m_dimension; ++i)
        {
            std::cout << ", " << subtree_model_config_.points_mean[i];
        }
        std::cout << std::endl;

        std::cout << "points_std: " << subtree_model_config_.points_std[0];
        for (uint64_t i = 1; i < m_dimension; ++i)
        {
            std::cout << ", " << subtree_model_config_.points_std[i];
        }
        std::cout << std::endl;

        std::cout << "geometric_properties_mean: " << subtree_model_config_.geometric_properties_mean[0];
        for (uint64_t i = 1; i < m_dimension; ++i)
        {
            std::cout << ", " << subtree_model_config_.geometric_properties_mean[i];
        }
        std::cout << std::endl;

        std::cout << "geometric_properties_std: " << subtree_model_config_.geometric_properties_std[0];
        for (uint64_t i = 1; i < m_dimension; ++i)
        {
            std::cout << ", " << subtree_model_config_.geometric_properties_std[i];
        }
        std::cout << std::endl;
    }

    // SubtreeInfo padding_subtree_info(std::move(subtree_infos.back()));
    // SubtreeInfo padding_subtree_info(subtree_infos.back());
    if (subtree_debug)
    {
        std::cout << "1. It is OK here." << std::endl;
        std::cout << "num_subtrees = " << num_subtrees << ", max_num_subtrees = " << subtree_model_config_.max_num_subtrees << std::endl;
    }

    if (num_subtrees > subtree_model_config_.max_num_subtrees)
    {
        subtree_infos.pop_back();
        if (level > 1)
        {
            std::sort(subtree_infos.begin(), subtree_infos.end(), SubtreeInfo::compare_internal);
        }
        else
        {
            std::sort(subtree_infos.begin(), subtree_infos.end(), SubtreeInfo::compare_leaf);
        }
        subtree_infos.resize(subtree_model_config_.max_num_subtrees);
        num_subtrees = subtree_model_config_.max_num_subtrees;
    }
    if (subtree_debug)
    {
        std::cout << "2. It is OK here." << std::endl;
    }

    std::vector<double> lbds_local;
    std::vector<double> diff_local;

    for (int64_t d = 0; d < dim; ++d)
    {
        lbds_local.emplace_back(m_nodeMBR.m_pLow[d]);
        diff_local.emplace_back(m_nodeMBR.m_pHigh[d] - m_nodeMBR.m_pLow[d]);
    }
    if (subtree_debug)
    {
        std::cout << "3. It is OK here." << std::endl;
    }

    input_subtree_context_rep_points.clear();
    for (std::vector<double> &point: subtree_context_rep_points)
    {
        std::vector<float> expanded_point;
        for (int64_t d = 0; d < dim; ++d)
        {
            double x = point[d];
            double expanded_x_local = (x - lbds_local[d]) / diff_local[d];
            double expanded_x_global = (x - lbds[d]) / (ubds[d] - lbds[d]);

            expanded_x_local = (expanded_x_local - subtree_model_config_.points_mean[2 * d]) / subtree_model_config_.points_std[2 * d];
            expanded_x_global = (expanded_x_global - subtree_model_config_.points_mean[2 * d + 1]) / subtree_model_config_.points_std[2 * d + 1];

            input_subtree_context_rep_points.emplace_back(static_cast<float>(expanded_x_local));
            input_subtree_context_rep_points.emplace_back(static_cast<float>(expanded_x_global));
        }
    }
    if (subtree_debug)
    {
        std::cout << "4. It is OK here." << std::endl;
    }

    input_subtree_candi_rep_points.clear();
    for (int64_t i = 0; i < num_subtrees; ++i)
    {
        SubtreeInfo &subtree_info = subtree_infos[i];
        std::vector< std::vector<double> > _points;
        get_region_boundary_points(subtree_info.bb, _points);

        for (std::vector<double> &point: _points)
        {
            for (int64_t d = 0; d < dim; ++d)
            {
                double x = point[d];
                double expanded_x_local = (x - lbds_local[d]) / diff_local[d];
                double expanded_x_global = (x - lbds[d]) / (ubds[d] - lbds[d]);

                expanded_x_local = (expanded_x_local - subtree_model_config_.points_mean[2 * d]) / subtree_model_config_.points_std[2 * d];
                expanded_x_global = (expanded_x_global - subtree_model_config_.points_mean[2 * d + 1]) / subtree_model_config_.points_std[2 * d + 1];


                input_subtree_candi_rep_points.emplace_back(static_cast<float>(expanded_x_local));
                input_subtree_candi_rep_points.emplace_back(static_cast<float>(expanded_x_global));
            }
        }
    }
    if (subtree_debug)
    {
        std::cout << "5. It is OK here." << std::endl;
    }

    if (num_subtrees < subtree_model_config_.max_num_subtrees)
    {
        std::vector<float> padding_expanded_point;
        SubtreeInfo &subtree_info = subtree_infos[num_subtrees];
        std::vector< std::vector<double> > _points;
        get_region_boundary_points(subtree_info.bb, _points);
        for (std::vector<double> &point: _points)
        {
            for (int64_t d = 0; d < dim; ++d)
            {
                double x = point[d];
                double expanded_x_local = (x - lbds_local[d]) / diff_local[d];
                double expanded_x_global = (x - lbds[d]) / (ubds[d] - lbds[d]);

                expanded_x_local = (expanded_x_local - subtree_model_config_.points_mean[2 * d]) / subtree_model_config_.points_std[2 * d];
                expanded_x_global = (expanded_x_global - subtree_model_config_.points_mean[2 * d + 1]) / subtree_model_config_.points_std[2 * d + 1];

                padding_expanded_point.emplace_back(static_cast<float>(expanded_x_local));
                padding_expanded_point.emplace_back(static_cast<float>(expanded_x_global));
            }
        }

        for (int64_t i = num_subtrees; i < subtree_model_config_.max_num_subtrees; ++i)
        {
            input_subtree_candi_rep_points.insert(input_subtree_candi_rep_points.end(), padding_expanded_point.begin(), padding_expanded_point.end());
        }
    }
    if (subtree_debug)
    {
        std::cout << "6. It is OK here." << std::endl;
    }

    input_subtree_geometric_properties.clear();
    for (int64_t i = 0; i < num_subtrees; ++i)
    {
        SubtreeInfo &subtree_info = subtree_infos[i];
        for (int64_t j = 0; j < subtree_model_config_.num_geometric_properties; ++j)
        {
            double x = (subtree_info.geometric_properties[j] - subtree_model_config_.geometric_properties_mean[j]) / subtree_model_config_.geometric_properties_std[j];
            input_subtree_geometric_properties.emplace_back(static_cast<float>(x));
        }
    }
    if (subtree_debug)
    {
        std::cout << "7. It is OK here." << std::endl;
    }
    if (num_subtrees < subtree_model_config_.max_num_subtrees)
    {
        std::vector<float> padding_geometric_properties;
        SubtreeInfo &subtree_info = subtree_infos[num_subtrees];
        for (int64_t j = 0; j < subtree_model_config_.num_geometric_properties; ++j)
        {
            double x = (subtree_info.geometric_properties[j] - subtree_model_config_.geometric_properties_mean[j]) / subtree_model_config_.geometric_properties_std[j];
            padding_geometric_properties.emplace_back(static_cast<float>(x));
        }
        for (int64_t i = num_subtrees; i < subtree_model_config_.max_num_subtrees; ++i)
        {
            input_subtree_geometric_properties.insert(input_subtree_geometric_properties.end(), padding_geometric_properties.begin(), padding_geometric_properties.end());
        }
    }


    if (subtree_debug)
    {
        std::cout << "8. It is OK here." << std::endl;
        std::cout << "input_subtree_context_rep_points.size = " << input_subtree_context_rep_points.size() << ". It should be " << subtree_model_config_.subtree_context_rep_points_shape[1] * subtree_model_config_.subtree_context_rep_points_shape[2] << std::endl;
        std::cout << "input_subtree_candi_rep_points.size = " << input_subtree_candi_rep_points.size() << ". It should be " << subtree_model_config_.subtree_candi_rep_points_shape[1] * subtree_model_config_.subtree_candi_rep_points_shape[2] << std::endl;
        std::cout << "input_subtree_geometric_properties.size = " << input_subtree_geometric_properties.size() << ". It should be " << subtree_model_config_.subtree_geometric_properties_shape[1] * subtree_model_config_.subtree_geometric_properties_shape[2] << std::endl;
    }

    // tensor_split_context_rep_points = torch::from_blob(input_subtree_split_context_rep_points.data(), split_context_rep_points_shape, cpu_options);
    // tensor_subtree_candi_rep_points = torch::from_blob(input_subtree_candi_rep_points.data(), subtree_candi_rep_points_shape, cpu_options);
    // tensor_subtree_geometric_properties = torch::from_blob(input_subtree_geometric_properties.data(), subtree_geometric_properties_shape, cpu_options);

    torch::Tensor tensor_subtree_context_rep_points = torch::from_blob(input_subtree_context_rep_points.data(), subtree_model_config_.subtree_context_rep_points_shape, cpu_options);
    torch::Tensor tensor_subtree_candi_rep_points = torch::from_blob(input_subtree_candi_rep_points.data(), subtree_model_config_.subtree_candi_rep_points_shape, cpu_options);
    torch::Tensor tensor_subtree_geometric_properties = torch::from_blob(input_subtree_geometric_properties.data(), subtree_model_config_.subtree_geometric_properties_shape, cpu_options);
    // std::cout << "11. It is OK here." << std::endl;
    torch::Tensor pred_scores_tensor = subtree_model.forward({tensor_subtree_context_rep_points.to(device), tensor_subtree_candi_rep_points.to(device), tensor_subtree_geometric_properties.to(device)}).toTensor();

    std::vector<float> pred_scores;
    for (int64_t j = 0; j < num_subtrees; ++j)
    {
        pred_scores.push_back(pred_scores_tensor.index({0, j}).item<float>());
    }
    int64_t pred_index = vector_utils::argmax(pred_scores);
    if (subtree_debug)
    {
        std::cout << "9. It is OK here." << std::endl;
        std::cout << "pred_index = " << pred_index << ", num_subtrees = " << num_subtrees << ", max_num_subtrees = " << subtree_model_config_.max_num_subtrees << std::endl;
    }

    return pred_index;
}
