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

#include <limits>

#include <spatialindex/SpatialIndex.h>
#include "RTree.h"
#include "Node.h"
#include "Leaf.h"
#include "Index.h"

#include "spatialindex/utils/print_utils.h"

using namespace SpatialIndex;
using namespace SpatialIndex::RTree;

Index::~Index()
= default;

Index::Index(SpatialIndex::RTree::RTree* pTree, id_type id, uint32_t level) : Node(
    pTree, id, level, pTree->m_indexCapacity)
{
}

// Index::Index(SpatialIndex::RTree::RTree* pTree, Node& ref): Node(pTree, ref)
// {
// }

NodePtr Index::chooseSubtree(const Region& mbr, uint32_t insertionLevel, std::stack<id_type>& pathBuffer)
{
    if (m_level == insertionLevel) return NodePtr(this, &(m_pTree->m_indexPool));

    pathBuffer.push(m_identifier);

    uint32_t child = 0;

    switch (m_pTree->m_treeVariant)
    {
    case RV_LINEAR:
    case RV_QUADRATIC:
        child = findLeastEnlargement(mbr);
        break;
    case RV_RSTAR:
    case RV_BEST_SPLIT:
    case RV_MODEL_SPLIT:
        if (m_level == 1)
        {
            // if this node points to leaves...
            child = findLeastOverlap(mbr);
        }
        else
        {
            child = findLeastEnlargement(mbr);
        }
    break;
    case RV_BEST_SUBTREE:
    case RV_BEST_BOTH:
        child = findBestSubtree(mbr, insertionLevel);

        // if (m_level == 1)
        // {
        //     // // if this node points to leaves...
        //     // child = findLeastOverlap(mbr);
        //     child = findBestSubtree(mbr, insertionLevel);
        // }
        // else
        // {
        //     child = findLeastEnlargement(mbr);
        // }

        break;
    case RV_MODEL_SUBTREE:
    case RV_MODEL_BOTH:
        child = findSubtree_w_model(mbr, insertionLevel);
        break;
    default:
        throw Tools::NotSupportedException("Index::chooseSubtree: Tree variant not supported.");
    }
    assert(child != std::numeric_limits<uint32_t>::max());

    NodePtr n = m_pTree->readNode(m_pIdentifier[child]);
    NodePtr ret = n->chooseSubtree(mbr, insertionLevel, pathBuffer);
    assert(n.unique());
    if (ret.get() == n.get()) n.relinquish();

    return ret;
}

NodePtr Index::findLeaf(const Region& mbr, id_type id, std::stack<id_type>& pathBuffer)
{
    pathBuffer.push(m_identifier);

    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        if (m_ptrMBR[cChild]->containsRegion(mbr))
        {
            NodePtr n = m_pTree->readNode(m_pIdentifier[cChild]);
            NodePtr l = n->findLeaf(mbr, id, pathBuffer);
            if (n.get() == l.get()) n.relinquish();
            if (l.get() != nullptr) return l;
        }
    }

    pathBuffer.pop();

    return NodePtr();
}

void Index::split(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id,
                              std::stack<id_type>& pathBuffer, NodePtr& ptrLeft, NodePtr& ptrRight)
{
    ++(m_pTree->m_stats.m_u64Splits);

    std::vector<uint32_t> g1, g2;

    switch (m_pTree->m_treeVariant)
    {
    case RV_LINEAR:
    case RV_QUADRATIC:
        rtreeSplit(dataLength, pData, mbr, id, g1, g2);
        break;
    case RV_RSTAR:
    case RV_BEST_SUBTREE:
    case RV_MODEL_SUBTREE:
        rstarSplit(dataLength, pData, mbr, id, g1, g2);
        break;
    case RV_BEST_SPLIT:
    case RV_BEST_BOTH:
        chooseBestSplit(dataLength, pData, mbr, id, pathBuffer, g1, g2);
        break;
    case RV_MODEL_SPLIT:
    case RV_MODEL_BOTH:
        chooseSplit_w_model(dataLength, pData, mbr, id, pathBuffer, g1, g2);
        break;
    default:
        throw Tools::NotSupportedException("Index::split: Tree variant not supported.");
    }

    ptrLeft = m_pTree->m_indexPool.acquire();
    ptrRight = m_pTree->m_indexPool.acquire();

    if (ptrLeft.get() == nullptr) ptrLeft = NodePtr(new Index(m_pTree, m_identifier, m_level), &(m_pTree->m_indexPool));
    if (ptrRight.get() == nullptr) ptrRight = NodePtr(new Index(m_pTree, -1, m_level), &(m_pTree->m_indexPool));

    ptrLeft->m_nodeMBR = m_pTree->m_infiniteRegion;
    ptrRight->m_nodeMBR = m_pTree->m_infiniteRegion;

    uint32_t cIndex;

    for (cIndex = 0; cIndex < g1.size(); ++cIndex)
    {
        ptrLeft->insertEntry(0, nullptr, *(m_ptrMBR[g1[cIndex]]), m_pIdentifier[g1[cIndex]]);
    }

    for (cIndex = 0; cIndex < g2.size(); ++cIndex)
    {
        ptrRight->insertEntry(0, nullptr, *(m_ptrMBR[g2[cIndex]]), m_pIdentifier[g2[cIndex]]);
    }
}



uint32_t Index::findLeastEnlargement_clone(const Region& r) const
{
    double area = std::numeric_limits<double>::infinity();
    uint32_t best = std::numeric_limits<uint32_t>::max();

    RegionPtr t = m_pTree->m_regionPool.acquire();

    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        m_ptrMBR[cChild]->getCombinedRegion(*t, r);

        double a = m_ptrMBR[cChild]->getArea();
        double enl = t->getArea() - a;

        if (enl < area)
        {
            area = enl;
            best = cChild;
        }
        else if (enl == area)
        {
            // this will rarely happen, so compute best area on the fly only
            // when necessary.
            if (enl == std::numeric_limits<double>::infinity()
                || a < m_ptrMBR[best]->getArea())
                best = cChild;
        }
    }

    return best;
}

uint32_t Index::findLeastOverlap_clone(const Region& r) const
{
    OverlapEntry** entries = new OverlapEntry*[m_children];

    double leastOverlap = std::numeric_limits<double>::max();
    double me = std::numeric_limits<double>::max();
    OverlapEntry* best = nullptr;

    // find combined region and enlargement of every entry and store it.
    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        try
        {
            entries[cChild] = new OverlapEntry();
        }
        catch (...)
        {
            for (uint32_t i = 0; i < cChild; ++i) delete entries[i];
            delete[] entries;
            throw;
        }

        entries[cChild]->m_index = cChild;
        entries[cChild]->m_original = m_ptrMBR[cChild];
        entries[cChild]->m_combined = m_pTree->m_regionPool.acquire();
        m_ptrMBR[cChild]->getCombinedRegion(*(entries[cChild]->m_combined), r);
        entries[cChild]->m_oa = entries[cChild]->m_original->getArea();
        entries[cChild]->m_ca = entries[cChild]->m_combined->getArea();
        entries[cChild]->m_enlargement = entries[cChild]->m_ca - entries[cChild]->m_oa;

        if (entries[cChild]->m_enlargement < me)
        {
            me = entries[cChild]->m_enlargement;
            best = entries[cChild];
        }
        else if (entries[cChild]->m_enlargement == me && entries[cChild]->m_oa < best->m_oa)
        {
            best = entries[cChild];
        }
    }

    if (me < -std::numeric_limits<double>::epsilon() || me > std::numeric_limits<double>::epsilon())
    {
        uint32_t cIterations;

        if (m_children > m_pTree->m_nearMinimumOverlapFactor)
        {
            // sort entries in increasing order of enlargement.
            ::qsort(entries, m_children,
                    sizeof(OverlapEntry*),
                    OverlapEntry::compareEntries);
            assert(entries[0]->m_enlargement <= entries[m_children - 1]->m_enlargement);

            cIterations = m_pTree->m_nearMinimumOverlapFactor;
        }
        else
        {
            cIterations = m_children;
        }

        // calculate overlap of most important original entries (near minimum overlap cost).
        for (uint32_t cIndex = 0; cIndex < cIterations; ++cIndex)
        {
            double dif = 0.0;
            OverlapEntry* e = entries[cIndex];

            for (uint32_t cChild = 0; cChild < m_children; ++cChild)
            {
                if (e->m_index != cChild)
                {
                    double f = e->m_combined->getIntersectingArea(*(m_ptrMBR[cChild]));
                    if (f != 0.0) dif += f - e->m_original->getIntersectingArea(*(m_ptrMBR[cChild]));
                }
            } // for (cChild)

            if (dif < leastOverlap)
            {
                leastOverlap = dif;
                best = entries[cIndex];
            }
            else if (dif == leastOverlap)
            {
                if (e->m_enlargement == best->m_enlargement)
                {
                    // keep the one with least area.
                    if (e->m_original->getArea() < best->m_original->getArea()) best = entries[cIndex];
                }
                else
                {
                    // keep the one with least enlargement.
                    if (e->m_enlargement < best->m_enlargement) best = entries[cIndex];
                }
            }
        } // for (cIndex)
    }

    uint32_t ret = best->m_index;

    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        delete entries[cChild];
    }
    delete[] entries;

    return ret;
}

uint32_t Index::findLeastEnlargement(const Region& r) const
{
    double area = std::numeric_limits<double>::infinity();
    uint32_t best = std::numeric_limits<uint32_t>::max();

    RegionPtr t = m_pTree->m_regionPool.acquire();

    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        m_ptrMBR[cChild]->getCombinedRegion(*t, r);

        double a = m_ptrMBR[cChild]->getArea();
        double enl = t->getArea() - a;

        if (enl < area)
        {
            area = enl;
            best = cChild;
        }
        else if (enl == area)
        {
            // this will rarely happen, so compute best area on the fly only
            // when necessary.
            if (enl == std::numeric_limits<double>::infinity()
                || a < m_ptrMBR[best]->getArea())
                best = cChild;
        }
    }

    return best;
}

uint32_t Index::findLeastOverlap(const Region& r) const
{
    OverlapEntry** entries = new OverlapEntry*[m_children];

    double leastOverlap = std::numeric_limits<double>::max();
    double me = std::numeric_limits<double>::max();
    OverlapEntry* best = nullptr;

    // find combined region and enlargement of every entry and store it.
    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        try
        {
            entries[cChild] = new OverlapEntry();
        }
        catch (...)
        {
            for (uint32_t i = 0; i < cChild; ++i) delete entries[i];
            delete[] entries;
            throw;
        }

        entries[cChild]->m_index = cChild;
        entries[cChild]->m_original = m_ptrMBR[cChild];
        entries[cChild]->m_combined = m_pTree->m_regionPool.acquire();
        m_ptrMBR[cChild]->getCombinedRegion(*(entries[cChild]->m_combined), r);
        entries[cChild]->m_oa = entries[cChild]->m_original->getArea();
        entries[cChild]->m_ca = entries[cChild]->m_combined->getArea();
        entries[cChild]->m_enlargement = entries[cChild]->m_ca - entries[cChild]->m_oa;

        if (entries[cChild]->m_enlargement < me)
        {
            me = entries[cChild]->m_enlargement;
            best = entries[cChild];
        }
        else if (entries[cChild]->m_enlargement == me && entries[cChild]->m_oa < best->m_oa)
        {
            best = entries[cChild];
        }
    }

    if (me < -std::numeric_limits<double>::epsilon() || me > std::numeric_limits<double>::epsilon())
    {
        uint32_t cIterations;

        if (m_children > m_pTree->m_nearMinimumOverlapFactor)
        {
            // sort entries in increasing order of enlargement.
            ::qsort(entries, m_children,
                    sizeof(OverlapEntry*),
                    OverlapEntry::compareEntries);
            assert(entries[0]->m_enlargement <= entries[m_children - 1]->m_enlargement);

            cIterations = m_pTree->m_nearMinimumOverlapFactor;
        }
        else
        {
            cIterations = m_children;
        }

        // calculate overlap of most important original entries (near minimum overlap cost).
        for (uint32_t cIndex = 0; cIndex < cIterations; ++cIndex)
        {
            double dif = 0.0;
            OverlapEntry* e = entries[cIndex];

            for (uint32_t cChild = 0; cChild < m_children; ++cChild)
            {
                if (e->m_index != cChild)
                {
                    double f = e->m_combined->getIntersectingArea(*(m_ptrMBR[cChild]));
                    if (f != 0.0) dif += f - e->m_original->getIntersectingArea(*(m_ptrMBR[cChild]));
                }
            } // for (cChild)

            if (dif < leastOverlap)
            {
                leastOverlap = dif;
                best = entries[cIndex];
            }
            else if (dif == leastOverlap)
            {
                if (e->m_enlargement == best->m_enlargement)
                {
                    // keep the one with least area.
                    if (e->m_original->getArea() < best->m_original->getArea()) best = entries[cIndex];
                }
                else
                {
                    // keep the one with least enlargement.
                    if (e->m_enlargement < best->m_enlargement) best = entries[cIndex];
                }
            }
        } // for (cIndex)
    }

    uint32_t ret = best->m_index;

    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        delete entries[cChild];
    }
    delete[] entries;

    return ret;
}


uint32_t Index::findLeastFanout(const Region& r) const
{
    OverlapEntry** entries = new OverlapEntry*[m_children];

    double leastOverlap = std::numeric_limits<double>::max();
    double me = std::numeric_limits<double>::max();
    OverlapEntry* best = nullptr;

    // find combined region and enlargement of every entry and store it.
    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        try
        {
            entries[cChild] = new OverlapEntry();
        }
        catch (...)
        {
            for (uint32_t i = 0; i < cChild; ++i) delete entries[i];
            delete[] entries;
            throw;
        }

        entries[cChild]->m_index = cChild;
        entries[cChild]->m_original = m_ptrMBR[cChild];
        entries[cChild]->m_combined = m_pTree->m_regionPool.acquire();
        m_ptrMBR[cChild]->getCombinedRegion(*(entries[cChild]->m_combined), r);
        entries[cChild]->m_oa = entries[cChild]->m_original->getArea();
        entries[cChild]->m_ca = entries[cChild]->m_combined->getArea();
        entries[cChild]->m_enlargement = entries[cChild]->m_ca - entries[cChild]->m_oa;

        NodePtr n = m_pTree->readNode(m_pIdentifier[cChild]);
        entries[cChild]->m_fanout = n->m_children;

        if (entries[cChild]->m_enlargement < me)
        {
            me = entries[cChild]->m_enlargement;
            best = entries[cChild];
        }
        // else if (entries[cChild]->m_enlargement == me)
        // {
        //     if (entries[cChild]->m_fanout < best->m_fanout)
        //     {
        //         best = entries[cChild];
        //     }
        //     else if (entries[cChild]->m_fanout == best->m_fanout && entries[cChild]->m_oa < best->m_oa)
        //     {
        //         best = entries[cChild];
        //     }
        // }

        else if (entries[cChild]->m_enlargement == me && entries[cChild]->m_oa < best->m_oa)
        {
            best = entries[cChild];
        }
    }

    if (me < -std::numeric_limits<double>::epsilon() || me > std::numeric_limits<double>::epsilon())
    {
        uint32_t cIterations;

        if (m_children > m_pTree->m_nearMinimumOverlapFactor)
        {
            // sort entries in increasing order of enlargement.
            ::qsort(entries, m_children,
                    sizeof(OverlapEntry*),
                    OverlapEntry::compareEntries);
            assert(entries[0]->m_enlargement <= entries[m_children - 1]->m_enlargement);

            cIterations = m_pTree->m_nearMinimumOverlapFactor;
        }
        else
        {
            cIterations = m_children;
        }

        // calculate overlap of most important original entries (near minimum overlap cost).
        for (uint32_t cIndex = 0; cIndex < cIterations; ++cIndex)
        {
            double dif = 0.0;
            OverlapEntry* e = entries[cIndex];

            for (uint32_t cChild = 0; cChild < m_children; ++cChild)
            {
                if (e->m_index != cChild)
                {
                    double f = e->m_combined->getIntersectingArea(*(m_ptrMBR[cChild]));
                    if (f != 0.0) dif += f - e->m_original->getIntersectingArea(*(m_ptrMBR[cChild]));
                }
            } // for (cChild)

            if (dif < leastOverlap)
            {
                leastOverlap = dif;
                best = entries[cIndex];
            }
            else if (dif == leastOverlap)
            {
                if (e->m_enlargement == best->m_enlargement)
                {
                    // keep the one with least area.
                    if (e->m_original->getArea() < best->m_original->getArea()) best = entries[cIndex];
                }
                else
                {
                    // keep the one with least enlargement.
                    if (e->m_enlargement < best->m_enlargement) best = entries[cIndex];
                }
            }
        } // for (cIndex)
    }

    uint32_t ret = best->m_index;

    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        delete entries[cChild];
    }
    delete[] entries;

    return ret;
}


void SpatialIndex::RTree::get_increased_bbs(const Region &original_bb, const Region &combined_bb, std::vector<Region> &increased_bbs)
{
    Region current = combined_bb;
    uint32_t dim = combined_bb.m_dimension;
    increased_bbs.clear();
    for (uint32_t axis = 0; axis < dim; ++axis) {
        if (original_bb.m_pLow[axis] > current.m_pLow[axis]) {
            // Add the portion of A that's to the left of B on this axis
            Region left = current;
            left.m_pHigh[axis] = std::min(left.m_pHigh[axis], original_bb.m_pLow[axis]);
            // if (left.isValid()) { // Do not need to check it
            increased_bbs.push_back(left);
            // }
            current.m_pLow[axis] = original_bb.m_pLow[axis];
        }

        if (original_bb.m_pHigh[axis] < current.m_pHigh[axis]) {
            // Add the portion of A that's to the right of B on this axis
            Region right = current;
            right.m_pLow[axis] = std::max(right.m_pLow[axis], original_bb.m_pHigh[axis]);
            // if (right.isValid()) {
            increased_bbs.push_back(right);
            // }
            current.m_pHigh[axis] = original_bb.m_pHigh[axis];
        }
        // if (!current.isValid()) {
        //     break;
        // }
    }
}

void Index::getSubtreeInfo(const Region& r, uint32_t insertionLevel, uint32_t cIndex, OverlapEntry *e, double increased_overlap, std::vector<SubtreeInfo> &subtree_infos) const
{
    subtree_infos.emplace_back();
    SubtreeInfo &subtree_info = subtree_infos.back();
    subtree_info.cIndex = cIndex;
    subtree_info.increased_area = e->m_enlargement;
    subtree_info.increased_margin = e->m_margin_enlargement;
    subtree_info.increased_overlap = increased_overlap;
    subtree_info.occupancy_rate = static_cast<double>(e->m_fanout) / static_cast<double>(m_capacity);
    subtree_info.bb = *(e->m_original);
    subtree_info.combined_bb = *(e->m_combined);

    std::vector< std::vector<Region> > path_increased_bbs;
    std::vector<Region> original_bbs;

    original_bbs.emplace_back(subtree_info.bb);
    path_increased_bbs.emplace_back();
    std::vector<Region> &increased_bbs_level_0 = path_increased_bbs.back();
    increased_bbs_level_0.emplace_back(subtree_info.combined_bb);
    // get_increased_bbs(subtree_info.bb, subtree_info.combined_bb, increased_bbs_level_0);

    uint64_t cost = calcChooseSubtreeCost_w_QueryTree(original_bbs, path_increased_bbs, subtree_info, m_pTree->queryTree);
    // uint64_t cost = calcChooseSubtreeCost_w_QueryTree_v2(original_bbs, path_increased_bbs, subtree_info, queryTree, queries);
    // uint64_t cost = calcChooseSubtreeCost_w_Queries(original_bbs, path_increased_bbs, queries);
    subtree_info.io_cost = cost;
    // io_cost_each_subtree.emplace_back(cost);
}

void Index::getSubtreeInfo_complex(const Region& r, uint32_t insertionLevel, OverlapEntry *e, double increased_overlap, std::vector<SubtreeInfo> &subtree_infos) const
{
    subtree_infos.emplace_back();
    SubtreeInfo &subtree_info = subtree_infos.back();
    subtree_info.increased_area = e->m_enlargement;
    subtree_info.increased_margin = e->m_margin_enlargement;
    subtree_info.increased_overlap = increased_overlap;
    subtree_info.occupancy_rate = static_cast<double>(e->m_fanout) / static_cast<double>(m_capacity);
    subtree_info.bb = *(e->m_original);
    subtree_info.combined_bb = *(e->m_combined);

    std::vector< std::vector<Region> > path_increased_bbs;
    std::vector<Region> original_bbs;

    original_bbs.emplace_back(subtree_info.bb);
    path_increased_bbs.emplace_back();
    std::vector<Region> &increased_bbs_level_0 = path_increased_bbs.back();
    // increased_bbs_level_0.emplace_back(subtree_info.combined_bb);
    get_increased_bbs(subtree_info.bb, subtree_info.combined_bb, increased_bbs_level_0);
    if (m_level > insertionLevel + 1)
    {
        uint32_t child = e->m_index;
        NodePtr n = m_pTree->readNode(m_pIdentifier[child]);
        uint32_t node_level = n->m_level;
        while (node_level > insertionLevel)
        {
            if (node_level == 1)
            {
                // if this node points to leaves...
                child = n->findLeastOverlap_clone(r);
            }
            else
            {
                child = n->findLeastEnlargement_clone(r);
            }

            assert(child != std::numeric_limits<uint32_t>::max());

            n = m_pTree->readNode(n->m_pIdentifier[child]);
            node_level = n->m_level;
            original_bbs.emplace_back(n->m_nodeMBR);
            Region combined_bb = n->m_nodeMBR;
            combined_bb.combineRegion(r);
            path_increased_bbs.emplace_back();
            std::vector<Region> &increased_bbs_a_level = path_increased_bbs.back();
            get_increased_bbs(n->m_nodeMBR, combined_bb, increased_bbs_a_level);
            // std::cout << "\tpath_increased_bbs.size = " << path_increased_bbs.size() << ", increased_bbs_a_level.size = " << increased_bbs_a_level.size() << std::endl;
        }
    }

    uint64_t cost = calcChooseSubtreeCost_w_QueryTree(original_bbs, path_increased_bbs, subtree_info, m_pTree->queryTree);
    // uint64_t cost = calcChooseSubtreeCost_w_QueryTree_v2(original_bbs, path_increased_bbs, subtree_info, queryTree, queries);
    // uint64_t cost = calcChooseSubtreeCost_w_Queries(original_bbs, path_increased_bbs, queries);
    subtree_info.io_cost = cost;
    // io_cost_each_subtree.emplace_back(cost);
}


void Index::getPaddingSubtreeInfo(const Region& r, std::vector<SubtreeInfo> &subtree_infos) const
{
    subtree_infos.emplace_back();
    SubtreeInfo &subtree_info = subtree_infos.back();
    subtree_info.cIndex = m_children;
    subtree_info.increased_area = std::numeric_limits<double>::max();
    subtree_info.increased_margin = std::numeric_limits<double>::max();
    subtree_info.increased_overlap = std::numeric_limits<double>::max();
    subtree_info.occupancy_rate = 1;
    subtree_info.bb = m_nodeMBR;
    subtree_info.combined_bb = m_nodeMBR;
    subtree_info.combined_bb.combineRegion(r);
    subtree_info.io_cost = std::numeric_limits<uint64_t>::max();

    // io_cost_each_subtree.emplace_back(std::numeric_limits<uint64_t>::max());
}


uint32_t Index::findBestSubtree(const Region& r, uint32_t insertionLevel) const
{
    OverlapEntry** entries = new OverlapEntry*[m_children];

    double leastOverlap = std::numeric_limits<double>::max();
    double me = std::numeric_limits<double>::max();
    OverlapEntry* best = nullptr;
    OverlapEntry* io_best = nullptr;

    // find combined region and enlargement of every entry and store it.
    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        try
        {
            entries[cChild] = new OverlapEntry();
        }
        catch (...)
        {
            for (uint32_t i = 0; i < cChild; ++i) delete entries[i];
            delete[] entries;
            throw;
        }

        entries[cChild]->m_index = cChild;
        entries[cChild]->m_original = m_ptrMBR[cChild];
        entries[cChild]->m_combined = m_pTree->m_regionPool.acquire();
        m_ptrMBR[cChild]->getCombinedRegion(*(entries[cChild]->m_combined), r);
        entries[cChild]->m_oa = entries[cChild]->m_original->getArea();
        entries[cChild]->m_ca = entries[cChild]->m_combined->getArea();
        entries[cChild]->m_enlargement = entries[cChild]->m_ca - entries[cChild]->m_oa;

        entries[cChild]->m_om = entries[cChild]->m_original->getMargin();
        entries[cChild]->m_cm = entries[cChild]->m_combined->getMargin();
        entries[cChild]->m_margin_enlargement = entries[cChild]->m_cm - entries[cChild]->m_om;

        NodePtr n = m_pTree->readNode(m_pIdentifier[cChild]);
        entries[cChild]->m_fanout = n->m_children;


        if (entries[cChild]->m_enlargement < me)
        {
            me = entries[cChild]->m_enlargement;
            best = entries[cChild];
        }
        else if (entries[cChild]->m_enlargement == me && entries[cChild]->m_oa < best->m_oa)
        {
            best = entries[cChild];
        }
    }

    // std::cout << "1. chosen = " << best->m_index << std::endl;

    if (me < -std::numeric_limits<double>::epsilon() || me > std::numeric_limits<double>::epsilon())
    {
        ++(m_pTree->num_find_best_subtree);
        uint32_t cIterations;

        if (m_children > m_pTree->m_nearMinimumOverlapFactor)
        {
            // sort entries in increasing order of enlargement.
            ::qsort(entries, m_children,
                    sizeof(OverlapEntry*),
                    OverlapEntry::compareEntries);
            assert(entries[0]->m_enlargement <= entries[m_children - 1]->m_enlargement);

            cIterations = m_pTree->m_nearMinimumOverlapFactor;
        }
        else
        {
            cIterations = m_children;
        }

        int64_t num_subtrees_threshold = m_pTree->num_subtrees_threshold;
        std::vector<SubtreeInfo> subtree_infos;
        std::vector<uint64_t> io_cost_each_subtree;

        // RTree *queryTree = m_pTree->queryTree;
        // std::vector<Region> queries;
        // queryTree->getOverlapMBRs(r, queries);

        // calculate overlap of most important original entries (near minimum overlap cost).
        uint32_t best_cIndex = cIterations;
        double anchor_dif = -1;
        for (uint32_t cIndex = 0; cIndex < cIterations; ++cIndex)
        {
            double dif = 0.0;
            OverlapEntry* e = entries[cIndex];

            for (uint32_t cChild = 0; cChild < m_children; ++cChild)
            {
                if (e->m_index != cChild)
                {
                    double f = e->m_combined->getIntersectingArea(*(m_ptrMBR[cChild]));
                    if (f != 0.0) dif += f - e->m_original->getIntersectingArea(*(m_ptrMBR[cChild]));
                }
            } // for (cChild)

            if (dif < leastOverlap)
            {
                leastOverlap = dif;
                best_cIndex = cIndex;
                best = entries[cIndex];
            }
            else if (dif == leastOverlap)
            {
                if (e->m_enlargement == best->m_enlargement)
                {
                    // keep the one with least area.
                    if (e->m_original->getArea() < best->m_original->getArea())
                    {
                        best_cIndex = cIndex;
                        best = entries[cIndex];
                    }
                }
                else
                {
                    // keep the one with least enlargement.
                    if (e->m_enlargement < best->m_enlargement)
                    {
                        best_cIndex = cIndex;
                        best = entries[cIndex];
                    }
                }
            }

            if (num_subtrees_threshold < 0 || (cIndex < num_subtrees_threshold - 1))
            {
                getSubtreeInfo(r, insertionLevel, cIndex, e, dif, subtree_infos);
            }
            else if (cIndex == num_subtrees_threshold - 1)
            {
                anchor_dif = dif;
            }
            // std::cout << "cIndex = " << cIndex << ", cost = " << cost << ", best_cost = " << best_cost << std::endl;

        } // for (cIndex)

        bool flag = false;
        if (num_subtrees_threshold > 0 && num_subtrees_threshold < cIterations)
        {
            if (best_cIndex < num_subtrees_threshold)
            {
                getSubtreeInfo(r, insertionLevel, num_subtrees_threshold - 1, entries[num_subtrees_threshold - 1], anchor_dif, subtree_infos);
            }
            else
            {
                getSubtreeInfo(r, insertionLevel, best_cIndex, entries[best_cIndex], leastOverlap, subtree_infos);
                flag = true;
            }
        }
        getPaddingSubtreeInfo(r, subtree_infos);
        std::vector<double> subtree_scores;
        std::vector< double > geometric_stats;
        get_subtree_geometric_properties(m_nodeMBR, subtree_infos, geometric_stats);
        int64_t chosen_index = get_subtree_scores(subtree_infos, geometric_stats, subtree_scores);

        // std::cout << "chosen_index: " << chosen_index << ", cIterations = " << cIterations << ", num_subtrees_threshold = " << num_subtrees_threshold << ", subtree_infos.size = " << subtree_infos.size() << std::endl;
        if (chosen_index < 0 || chosen_index >= subtree_infos.size() - 1)
        {
            throw Tools::BasicException("In Index::findBestSubtree, chosen_index = " + std::to_string(chosen_index) + ", subtree_infos.size = " + std::to_string(subtree_infos.size()) + ".");
        }
        uint32_t best_index = subtree_infos[chosen_index].cIndex;
        best = entries[best_index];
        
        if (isSubtreeRecordingTurnedOn())
        {
            m_pTree->writeASubtreeRecord(r, m_nodeMBR, subtree_infos, subtree_scores, m_level);
        }
    }

    uint32_t ret = best->m_index;

    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        delete entries[cChild];
    }
    delete[] entries;

    return ret;

}


uint32_t Index::findSubtree_w_model(const Region& r, uint32_t insertionLevel) const
{
    OverlapEntry** entries = new OverlapEntry*[m_children];

    double leastOverlap = std::numeric_limits<double>::max();
    double me = std::numeric_limits<double>::max();
    OverlapEntry* best = nullptr;
    OverlapEntry* io_best = nullptr;

    // find combined region and enlargement of every entry and store it.
    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        try
        {
            entries[cChild] = new OverlapEntry();
        }
        catch (...)
        {
            for (uint32_t i = 0; i < cChild; ++i) delete entries[i];
            delete[] entries;
            throw;
        }

        entries[cChild]->m_index = cChild;
        entries[cChild]->m_original = m_ptrMBR[cChild];
        entries[cChild]->m_combined = m_pTree->m_regionPool.acquire();
        m_ptrMBR[cChild]->getCombinedRegion(*(entries[cChild]->m_combined), r);
        entries[cChild]->m_oa = entries[cChild]->m_original->getArea();
        entries[cChild]->m_ca = entries[cChild]->m_combined->getArea();
        entries[cChild]->m_enlargement = entries[cChild]->m_ca - entries[cChild]->m_oa;

        entries[cChild]->m_om = entries[cChild]->m_original->getMargin();
        entries[cChild]->m_cm = entries[cChild]->m_combined->getMargin();
        entries[cChild]->m_margin_enlargement = entries[cChild]->m_cm - entries[cChild]->m_om;

        NodePtr n = m_pTree->readNode(m_pIdentifier[cChild]);
        entries[cChild]->m_fanout = n->m_children;

        if (entries[cChild]->m_enlargement < me)
        {
            me = entries[cChild]->m_enlargement;
            best = entries[cChild];
        }
        else if (entries[cChild]->m_enlargement == me && entries[cChild]->m_oa < best->m_oa)
        {
            best = entries[cChild];
        }
    }

    // std::cout << "1. chosen = " << best->m_index << std::endl;

    if (me < -std::numeric_limits<double>::epsilon() || me > std::numeric_limits<double>::epsilon())
    {
        uint32_t cIterations;

        if (m_children > m_pTree->m_nearMinimumOverlapFactor)
        {
            // sort entries in increasing order of enlargement.
            ::qsort(entries, m_children,
                    sizeof(OverlapEntry*),
                    OverlapEntry::compareEntries);
            assert(entries[0]->m_enlargement <= entries[m_children - 1]->m_enlargement);

            cIterations = m_pTree->m_nearMinimumOverlapFactor;
        }
        else
        {
            cIterations = m_children;
        }

        int64_t num_subtrees_threshold = m_pTree->num_subtrees_threshold;
        std::vector<SubtreeInfo> subtree_infos;

        // RTree *queryTree = m_pTree->queryTree;
        // std::vector<Region> queries;
        // queryTree->getOverlapMBRs(r, queries);

        // calculate overlap of most important original entries (near minimum overlap cost).
        uint32_t best_cIndex = cIterations;
        double anchor_dif = -1;
        for (uint32_t cIndex = 0; cIndex < cIterations; ++cIndex)
        {
            double dif = 0.0;
            OverlapEntry* e = entries[cIndex];

            for (uint32_t cChild = 0; cChild < m_children; ++cChild)
            {
                if (e->m_index != cChild)
                {
                    double f = e->m_combined->getIntersectingArea(*(m_ptrMBR[cChild]));
                    if (f != 0.0) dif += f - e->m_original->getIntersectingArea(*(m_ptrMBR[cChild]));
                }
            } // for (cChild)

            if (dif < leastOverlap)
            {
                leastOverlap = dif;
                best_cIndex = cIndex;
                best = entries[cIndex];
            }
            else if (dif == leastOverlap)
            {
                if (e->m_enlargement == best->m_enlargement)
                {
                    // keep the one with least area.
                    if (e->m_original->getArea() < best->m_original->getArea())
                    {
                        best_cIndex = cIndex;
                        best = entries[cIndex];
                    }
                }
                else
                {
                    // keep the one with least enlargement.
                    if (e->m_enlargement < best->m_enlargement)
                    {
                        best_cIndex = cIndex;
                        best = entries[cIndex];
                    }
                }
            }
            getSubtreeInfo(r, insertionLevel, cIndex, e, dif, subtree_infos);
            // std::cout << "cIndex = " << cIndex << ", cost = " << cost << ", best_cost = " << best_cost << std::endl;
        } // for (cIndex)

        if (num_subtrees_threshold > 0 && num_subtrees_threshold < cIterations)
        {
            if (m_level > 1)
            {
                std::sort(subtree_infos.begin(), subtree_infos.end(), SubtreeInfo::compare_internal);
            }
            else
            {
                std::sort(subtree_infos.begin(), subtree_infos.end(), SubtreeInfo::compare_leaf);
            }
            subtree_infos.resize(num_subtrees_threshold);
        }
        getPaddingSubtreeInfo(r, subtree_infos);
        std::vector<double> subtree_scores;
        std::vector< double > geometric_stats;
        get_subtree_geometric_properties(m_nodeMBR, subtree_infos, geometric_stats);
        int64_t pred_index = m_pTree->subtreePredict(r, m_nodeMBR, subtree_infos, m_level);
        // std::cout << "best_index: " << best_index << ", cIterations = " << cIterations << ", num_subtrees_threshold = " << num_subtrees_threshold << ", subtree_infos.size = " << subtree_infos.size() << std::endl;
        if (pred_index < 0 || pred_index >= subtree_infos.size() - 1)
        {
            throw Tools::BasicException("In Index::findSubtree_w_model, pred_index = " + std::to_string(pred_index) + ", subtree_infos.size = " + std::to_string(subtree_infos.size()) + ".");
        }
        uint32_t best_index = subtree_infos[pred_index].cIndex;
        best = entries[best_index];

        if (isSubtreeRecordingTurnedOn())
        {
            m_pTree->writeASubtreeRecord(r, m_nodeMBR, subtree_infos, subtree_scores, m_level);
        }
    }

    uint32_t ret = best->m_index;

    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        delete entries[cChild];
    }
    delete[] entries;

    return ret;

}



uint64_t Index::calcChooseSubtreeCost_w_QueryTree(std::vector<Region> &original_bbs, std::vector< std::vector<Region> > &path_increased_bbs, SubtreeInfo &subtree_infos, RTree *query_tree)
{
    uint64_t restricted_cost = 0;
    uint64_t all_cost = 0;
    uint32_t path_len = original_bbs.size();

    for (uint32_t i = 0; i < path_len; ++i)
    {
        Region &original_bb = original_bbs[i];
        std::vector<Region> &path_increased_bbs_a_level = path_increased_bbs[i];
        std::unordered_set<id_type> mbr_ids;

        for (Region &bb: path_increased_bbs_a_level)
        {
            query_tree->getRestrictedOverlapMBRIDs(bb, original_bb, mbr_ids);
            // cost += query_tree->countRestrictedOverlapMBRs(bb, original_bb);
        }
        restricted_cost += mbr_ids.size();
        break;
    }

    return restricted_cost;
}

uint64_t Index::calcChooseSubtreeCost_w_QueryTree_v2(std::vector<Region> &original_bbs, std::vector< std::vector<Region> > &path_increased_bbs, SubtreeInfo &subtree_infos, RTree *query_tree, std::vector<Region> &queries)
{
    Region &original_bb0 = original_bbs[0];
    std::vector<Region> &path_increased_bbs_level_0 = path_increased_bbs[0];
    std::unordered_set<id_type> mbr_ids;
    for (Region &bb: path_increased_bbs_level_0)
    {
        query_tree->getRestrictedOverlapMBRIDs(bb, original_bb0, mbr_ids);
    }
    uint64_t cost = mbr_ids.size();
    uint32_t path_len = original_bbs.size();

    for (uint32_t i = 1; i < path_len; ++i)
    {
        Region &original_bb = original_bbs[i];
        std::vector<Region> &path_increased_bbs_a_level = path_increased_bbs[i];
        for (Region &q: queries)
        {
            if (!original_bb.intersectsRegion(q))
            {
                for (Region &bb: path_increased_bbs_a_level)
                {
                    if (bb.intersectsRegion(q))
                    {
                        ++cost;
                        break;
                    }
                }
            }
        }
    }

    return cost;
}

uint64_t Index::calcChooseSubtreeCost_w_Queries(std::vector<Region> &original_bbs, std::vector< std::vector<Region> > &path_increased_bbs, std::vector<Region> &queries)
{
    uint64_t cost = 0;
    uint32_t path_len = original_bbs.size();
    for (uint32_t i = 0; i < path_len; ++i)
    {
        Region &original_bb = original_bbs[i];
        std::vector<Region> &path_increased_bbs_a_level = path_increased_bbs[i];
        for (Region &q: queries)
        {
            if (!original_bb.intersectsRegion(q))
            {
                for (Region &bb: path_increased_bbs_a_level)
                {
                    if (bb.intersectsRegion(q))
                    {
                        ++cost;
                        break;
                    }
                }
            }
        }
        break;
    }
    return cost;
}

void SpatialIndex::RTree::get_subtree_geometric_properties(const Region &parent_mbr, std::vector< SubtreeInfo > &subtree_infos, std::vector< double > &geometric_stats)
{
    uint64_t num_subtrees = subtree_infos.size() - 1;
    double min_increased_overlap = std::numeric_limits<double>::max();
    double min_increased_area = std::numeric_limits<double>::max();
    double min_increased_margin = std::numeric_limits<double>::max();

    double max_increased_overlap = std::numeric_limits<double>::min();
    double max_increased_area = std::numeric_limits<double>::min();
    double max_increased_margin = std::numeric_limits<double>::min();

    double avg_increased_area = 0;
    double avg_increased_margin = 0;
    double avg_increased_overlap = 0;


    for (uint64_t s = 0; s < num_subtrees; ++s)
    {
        SubtreeInfo &subtree_info = subtree_infos[s];

        if (min_increased_overlap > subtree_info.increased_overlap)
        {
            min_increased_overlap = subtree_info.increased_overlap;
        }
        if (max_increased_overlap < subtree_info.increased_overlap)
        {
            max_increased_overlap = subtree_info.increased_overlap;
        }


        if (min_increased_area > subtree_info.increased_area)
        {
            min_increased_area = subtree_info.increased_area;
        }

        if (max_increased_area < subtree_info.increased_area)
        {
            max_increased_area = subtree_info.increased_area;
        }


        if (min_increased_margin > subtree_info.increased_margin)
        {
            min_increased_margin = subtree_info.increased_margin;
        }

        if (max_increased_margin < subtree_info.increased_margin)
        {
            max_increased_margin = subtree_info.increased_margin;
        }


        avg_increased_area += subtree_info.increased_area;
        avg_increased_margin += subtree_info.increased_margin;
        avg_increased_overlap += subtree_info.increased_overlap;
    }
    avg_increased_area /= (static_cast<double>(num_subtrees));
    avg_increased_overlap /= (static_cast<double>(num_subtrees));
    avg_increased_margin /= (static_cast<double>(num_subtrees));

    std::vector<double> _stats = {min_increased_overlap, min_increased_area, min_increased_margin,
        max_increased_overlap, max_increased_area, max_increased_margin,
        avg_increased_overlap, avg_increased_area, avg_increased_margin};
    geometric_stats = std::move(_stats);

    SubtreeInfo &padding_subtree_info = subtree_infos[num_subtrees];

    double parent_area = parent_mbr.getArea();
    double parent_margin = parent_mbr.getMargin();

    for (uint64_t s = 0; s < num_subtrees; ++s)
    {
        SubtreeInfo &subtree_info = subtree_infos[s];
        std::vector<double> &geometric_properties = subtree_info.geometric_properties;
        double overlap_feature = 0;
        if (subtree_info.increased_overlap > 0)
        {
            overlap_feature = subtree_info.increased_overlap / max_increased_overlap;
        }
        double area_feature = 0;
        if (subtree_info.increased_area > 0)
        {
            area_feature = subtree_info.increased_area / parent_area;
        }
        double margin_feature = 0;
        if (subtree_info.increased_margin > 0)
        {
            margin_feature = subtree_info.increased_margin / parent_margin;
        }
        geometric_properties.emplace_back(overlap_feature);
        geometric_properties.emplace_back(area_feature);
        geometric_properties.emplace_back(margin_feature);
        geometric_properties.emplace_back(subtree_info.occupancy_rate);
        // if (geometric_properties.size() != SubtreeInfo::get_num_geometric_properties())
        // {
        //     std::cout << "Error! geometric_properties.size() " << geometric_properties.size() << std::endl;
        //     throw Tools::BasicException("In RTree::get_subtree_geometric_properties, geometric_properties.size() != SubtreeInfo::get_num_geometric_properties().");
        // }
    }
    std::vector<double> &geometric_properties = padding_subtree_info.geometric_properties;
    geometric_properties.emplace_back(2.0);
    double area_feature = 2 * max_increased_area / parent_area;
    double margin_feature = 2 * max_increased_margin / parent_margin;
    geometric_properties.emplace_back(area_feature);
    geometric_properties.emplace_back(margin_feature);
    geometric_properties.emplace_back(1);
}

int64_t SpatialIndex::RTree::get_subtree_scores(std::vector< SubtreeInfo > &subtree_infos, std::vector< double > &geometric_stats, std::vector< double > &subtree_scores, bool print_info)
{
    uint64_t num_subtrees = subtree_infos.size() - 1;
    uint64_t best_io = std::numeric_limits<uint64_t>::max();
    uint64_t worst_io = 0;

    for (uint64_t s = 0; s < num_subtrees; ++s)
    {
        if (subtree_infos[s].io_cost < best_io)
        {
            best_io = subtree_infos[s].io_cost;
        }
        if (subtree_infos[s].io_cost > worst_io)
        {
            worst_io = subtree_infos[s].io_cost;
        }
    }
    if (print_info)
    {
        std::cout << "In get_subtree_scores: 1. worst_io = " << worst_io << ", num_subtrees = " << num_subtrees << std::endl;
    }

    subtree_scores.clear();
    std::vector< double > split_likelihoods;

    // std::vector<double> _stats = {min_overlap, min_area, min_margin, min_total_area, min_total_margin,
    //     max_overlap, max_area, max_margin, max_total_area, max_total_margin,
    //     avg_area, avg_margin, avg_total_area, avg_total_margin};

    double min_increased_overlap = geometric_stats[0];
    double min_increased_area = geometric_stats[1];
    double min_increased_margin = geometric_stats[2];
    double max_increased_overlap = geometric_stats[3];
    double max_increased_area = geometric_stats[4];
    double max_increased_margin = geometric_stats[5];
    double avg_increased_overlap = geometric_stats[6];
    double avg_increased_area = geometric_stats[7];
    double avg_increased_margin = geometric_stats[8];

    double avg_score = 0;
    if (worst_io > 0)
    {
        for (uint64_t s = 0; s < num_subtrees; ++s)
        {
            SubtreeInfo &subtree_info = subtree_infos[s];
            double score = 0;
            if (best_io > 0)
            {
                score = static_cast<double>(best_io) / static_cast<double>(subtree_infos[s].io_cost);
            }
            else
            {
                if (subtree_infos[s].io_cost == 0)
                {
                    score = 1;
                }
                else
                {
                    score = 0;
                }
            }
            avg_score += score;

            subtree_scores.emplace_back(score);
        }
    }

    SubtreeInfo &padding_subtree_info = subtree_infos[num_subtrees];
    if (worst_io > 0)
    {
        subtree_scores.emplace_back(0);
        avg_score /= static_cast<double>(num_subtrees);
    }

    std::vector< double > subtree_geometric_scores;
    double area_weight = 0.8;
    double overlap_weight = 0.1;
    double margin_weight = 0.05;
    double occupancy_weight = 0.05;

    double avg_geometric_score = 0;

    for (uint64_t s = 0; s < num_subtrees; ++s)
    {
        SubtreeInfo &subtree_info = subtree_infos[s];
        double overlap_score = 1;
        if (subtree_info.increased_overlap > 0)
        {
            overlap_score = 1.0 - subtree_info.increased_overlap / max_increased_overlap;
        }
        double area_score = 1;
        if (subtree_info.increased_area > 0)
        {
            area_score = 1.0 - subtree_info.increased_area / max_increased_area;
        }
        double margin_score = 1;
        if (subtree_info.increased_margin > 0)
        {
            margin_score = 1.0 - subtree_info.increased_margin / max_increased_margin;
        }
        double score = overlap_score * overlap_weight +
            area_score * area_weight + margin_score * margin_weight + (1 - subtree_info.occupancy_rate) * occupancy_weight;

        // double score = overlap_score * overlap_weight +
        //     area_score * area_weight + margin_score * margin_weight;

        avg_geometric_score += score;
        subtree_geometric_scores.emplace_back(score);

        if (print_info)
        {
            std::cout << "s = " << s << ", increased_overlap = " << subtree_info.increased_overlap << ", max_increased_overlap = " << max_increased_overlap << std::endl;
            std::cout << "\t" << "increased_area = " << subtree_info.increased_area << ", max_increased_area = " << max_increased_area << std::endl;
            std::cout << "\t" << "increased_margin = " << subtree_info.increased_margin << ", max_increased_margin = " << max_increased_margin << std::endl;
            std::cout << "(" << subtree_info.bb.m_pLow[0] << ", " << subtree_info.bb.m_pLow[1] << "), (" << subtree_info.bb.m_pHigh[0] << ", " << subtree_info.bb.m_pHigh[1] << ")" << std::endl;
        }
    }
    double geometric_score_sum = avg_geometric_score;
    avg_geometric_score /= static_cast<double>(num_subtrees);

    subtree_geometric_scores.emplace_back(0);

    // std::cout << "worst_io = " << worst_io << ", avg_score = " << avg_score << std::endl;

    if (worst_io == 0)
    {
        if (geometric_score_sum < std::numeric_limits<double>::epsilon() && geometric_score_sum > -std::numeric_limits<double>::epsilon())
        {
            for (uint64_t s = 0; s < num_subtrees; ++s)
            {
                subtree_geometric_scores[s] = 1;
            }
            avg_geometric_score = 1;
        }
        subtree_scores = std::move(subtree_geometric_scores);
        avg_score = avg_geometric_score;

        if (print_info)
        {
            print_utils::print_vec(subtree_scores, "subtree_scores");
        }
    }
    else
    {
        double score_sum = 0;
        double geometric_weight = 0.05;
        if (print_info)
        {
            print_utils::print_vec(subtree_scores, "1.subtree_scores");
        }
        if (print_info)
        {
            print_utils::print_vec(subtree_geometric_scores, "1.subtree_geometric_scores");
        }
        for (uint64_t s = 0; s < num_subtrees; ++s)
        {
            subtree_scores[s] += subtree_geometric_scores[s] * geometric_weight * avg_score;
            score_sum += subtree_scores[s];
        }
        if (print_info)
        {
            print_utils::print_vec(subtree_scores, "subtree_scores");
        }
        avg_score = score_sum / static_cast<double>(num_subtrees);
    }

    int64_t chosen = -1;
    double max_score = 0;


    double ratio = avg_score / 0.5;

    for (uint64_t s = 0; s <= num_subtrees; ++s)
    {
        double score = subtree_scores[s];
        score /= ratio;
        if (score > max_score)
        {
            chosen = s;
            max_score = score;
        }
        subtree_scores[s] = score;
    }

    return chosen;
}

void Index::adjustTree(Node* n, std::stack<id_type>& pathBuffer, bool force)
{
    ++(m_pTree->m_stats.m_u64Adjustments);

    // find entry pointing to old node;
    uint32_t child;
    for (child = 0; child < m_children; ++child)
    {
        if (m_pIdentifier[child] == n->m_identifier) break;
    }

    // MBR needs recalculation if either:
    //   1. the NEW child MBR is not contained.
    //   2. the OLD child MBR is touching.
    bool bContained = m_nodeMBR.containsRegion(n->m_nodeMBR);
    bool bTouches = m_nodeMBR.touchesRegion(*(m_ptrMBR[child]));
    bool bRecompute = (!bContained || (bTouches && m_pTree->m_bTightMBRs));

    *(m_ptrMBR[child]) = n->m_nodeMBR;

    if (bRecompute || force)
    {
        for (uint32_t cDim = 0; cDim < m_nodeMBR.m_dimension; ++cDim)
        {
            m_nodeMBR.m_pLow[cDim] = std::numeric_limits<double>::max();
            m_nodeMBR.m_pHigh[cDim] = -std::numeric_limits<double>::max();

            for (uint32_t cChild = 0; cChild < m_children; ++cChild)
            {
                m_nodeMBR.m_pLow[cDim] = std::min(m_nodeMBR.m_pLow[cDim], m_ptrMBR[cChild]->m_pLow[cDim]);
                m_nodeMBR.m_pHigh[cDim] = std::max(m_nodeMBR.m_pHigh[cDim], m_ptrMBR[cChild]->m_pHigh[cDim]);
            }
        }
    }

    m_pTree->writeNode(this);

    if ((bRecompute || force) && (!pathBuffer.empty()))
    {
        id_type cParent = pathBuffer.top();
        pathBuffer.pop();
        NodePtr ptrN = m_pTree->readNode(cParent);
        Index* p = static_cast<Index*>(ptrN.get());
        p->adjustTree(this, pathBuffer, force);
    }
}

void Index::adjustTree(Node* n1, Node* n2, std::stack<id_type>& pathBuffer, uint8_t* overflowTable)
{
    ++(m_pTree->m_stats.m_u64Adjustments);

    // find entry pointing to old node;
    uint32_t child;
    for (child = 0; child < m_children; ++child)
    {
        if (m_pIdentifier[child] == n1->m_identifier) break;
    }

    // MBR needs recalculation if either:
    //   1. either child MBR is not contained.
    //   2. the OLD child MBR is touching.
    bool bContained1 = m_nodeMBR.containsRegion(n1->m_nodeMBR);
    bool bContained2 = m_nodeMBR.containsRegion(n2->m_nodeMBR);
    bool bContained = bContained1 && bContained2;
    bool bTouches = m_nodeMBR.touchesRegion(*(m_ptrMBR[child]));
    bool bRecompute = (!bContained || (bTouches && m_pTree->m_bTightMBRs));

    *(m_ptrMBR[child]) = n1->m_nodeMBR;

    if (bRecompute)
    {
        for (uint32_t cDim = 0; cDim < m_nodeMBR.m_dimension; ++cDim)
        {
            m_nodeMBR.m_pLow[cDim] = std::numeric_limits<double>::max();
            m_nodeMBR.m_pHigh[cDim] = -std::numeric_limits<double>::max();

            for (uint32_t cChild = 0; cChild < m_children; ++cChild)
            {
                m_nodeMBR.m_pLow[cDim] = std::min(m_nodeMBR.m_pLow[cDim], m_ptrMBR[cChild]->m_pLow[cDim]);
                m_nodeMBR.m_pHigh[cDim] = std::max(m_nodeMBR.m_pHigh[cDim], m_ptrMBR[cChild]->m_pHigh[cDim]);
            }
        }
    }

    // No write necessary here. insertData will write the node if needed.
    //m_pTree->writeNode(this);

    bool bAdjusted = insertData(0, nullptr, n2->m_nodeMBR, n2->m_identifier, pathBuffer, overflowTable);

    // if n2 is contained in the node and there was no split or reinsert,
    // we need to adjust only if recalculation took place.
    // In all other cases insertData above took care of adjustment.
    if ((!bAdjusted) && bRecompute && (!pathBuffer.empty()))
    {
        id_type cParent = pathBuffer.top();
        pathBuffer.pop();
        NodePtr ptrN = m_pTree->readNode(cParent);
        Index* p = static_cast<Index*>(ptrN.get());
        p->adjustTree(this, pathBuffer);
    }
}

