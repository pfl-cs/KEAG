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

#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"

#include <cmath>
#include <unordered_set>


namespace SpatialIndex
{
    namespace RTree
    {
        class RTree;
        class Leaf;
        class Index;
        class Node;

        typedef Tools::PoolPointer<Node> NodePtr;
        typedef std::vector<uint32_t> splitGroup;

        class SubtreeInfo
        {
        public:
            uint32_t cIndex;
            double increased_overlap;
            double increased_margin;
            double increased_area;

            double occupancy_rate;

            Region combined_bb;
            Region bb;

            double score = 0;
            uint64_t io_cost;
            std::vector< std::vector<double> > candi_rep_points;
            std::vector<double> geometric_properties;

            static int32_t get_num_geometric_properties()
            {
                return 4;
            }


            static bool compare_leaf(const SubtreeInfo& si1, const SubtreeInfo& si2)
            {
                if (si1.increased_overlap < si2.increased_overlap) return true;
                if (si2.increased_overlap < si1.increased_overlap) return false;
                if (si1.increased_area < si2.increased_area) return true;
                if (si2.increased_area < si1.increased_area) return false;
                // if (si1.increased_margin < si2.increased_margin) return true;
                // if (si2.increased_margin < si1.increased_margin) return false;
                return si1.increased_margin < si2.increased_margin;
                // return si1.increased_overlap < si2.increased_overlap;
            }

            static bool compare_internal(const SubtreeInfo& si1, const SubtreeInfo& si2)
            {
                if (si1.increased_area < si2.increased_area) return true;
                if (si2.increased_area < si1.increased_area) return false;
                if (si1.increased_overlap < si2.increased_overlap) return true;
                if (si2.increased_overlap < si1.increased_overlap) return false;
                // if (si1.increased_margin < si2.increased_margin) return true;
                // if (si2.increased_margin < si1.increased_margin) return false;
                return si1.increased_margin < si2.increased_margin;
                // return si1.increased_overlap < si2.increased_overlap;
            }

            SubtreeInfo()
            {
                // margin = -1;
            }

            // SplitInfo& operator=(const SplitInfo &r)
            SubtreeInfo(const SubtreeInfo &r)
            {
                cIndex = r.cIndex;
                io_cost = r.io_cost;
                increased_area = r.increased_area;
                increased_margin = r.increased_margin;
                increased_overlap = r.increased_overlap;
                occupancy_rate = r.occupancy_rate;
                score = r.score;

                combined_bb = r.combined_bb;
                bb = r.bb;
                candi_rep_points = r.candi_rep_points;
                geometric_properties = r.geometric_properties;
            }

            SubtreeInfo(const SubtreeInfo &&r) noexcept
            {
                cIndex = r.cIndex;
                io_cost = r.io_cost;
                increased_area = r.increased_area;
                increased_margin = r.increased_margin;
                increased_overlap = r.increased_overlap;
                occupancy_rate = r.occupancy_rate;
                score = r.score;

                combined_bb = r.combined_bb;
                bb = r.bb;
                candi_rep_points = std::move(r.candi_rep_points);
                geometric_properties = std::move(r.geometric_properties);
            }

            SubtreeInfo & operator=(const SubtreeInfo &r)
            {
                if(this != &r)
                {
                    cIndex = r.cIndex;
                    io_cost = r.io_cost;
                    increased_area = r.increased_area;
                    increased_margin = r.increased_margin;
                    increased_overlap = r.increased_overlap;
                    occupancy_rate = r.occupancy_rate;
                    score = r.score;

                    combined_bb = r.combined_bb;
                    bb = r.bb;
                    candi_rep_points = r.candi_rep_points;
                    geometric_properties = r.geometric_properties;
                }

                return *this;
            }

            SubtreeInfo & operator=(const SubtreeInfo &&r) noexcept
            {
                cIndex = r.cIndex;
                io_cost = r.io_cost;
                increased_area = r.increased_area;
                increased_margin = r.increased_margin;
                increased_overlap = r.increased_overlap;
                occupancy_rate = r.occupancy_rate;
                score = r.score;

                combined_bb = r.combined_bb;
                bb = r.bb;
                candi_rep_points = std::move(r.candi_rep_points);
                geometric_properties = std::move(r.geometric_properties);
                return *this;
            }

            void get_subtree_candi_rep_points()
            {
                uint64_t dim = bb.m_dimension;
                uint64_t n_each_dim = 2;
                uint64_t num_points = static_cast<uint64_t>(pow(n_each_dim, dim) + 0.5);
                std::vector< std::vector<double> > vecs{ std::vector<double> (bb.m_pLow, bb.m_pLow + dim), std::vector<double> (bb.m_pHigh, bb.m_pHigh + dim) };

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
                    candi_rep_points.emplace_back(std::move(point));
                    // candi_rep_points.emplace_back(point);
                }
            }
        };


        class SplitInfo
        {
        public:
            splitGroup group1;
            splitGroup group2;

            uint32_t splitAxis;
            uint32_t sortOrder;
            double overlap;
            double margin;
            double area;

            double bb1_margin;
            double bb2_margin;
            double bb1_area;
            double bb2_area;
            // double side_len;
            double increased_area;
            Region bb1;
            Region bb2;

            std::vector< std::vector<double> > candi_rep_points;
            double score = 0;
            bool from_rstar;

            std::vector<double> geometric_properties;

            static int32_t get_num_geometric_properties()
            {
                return 5;
            }


            static bool compare(const SplitInfo& si1, const SplitInfo& si2)
            {
                if (si1.area < si2.area) return true;
                if (si2.area < si1.area) return false;
                // if (si1.overlap < si2.overlap) return true;
                // if (si2.overlap < si1.overlap) return false;
                if (si1.margin < si2.margin) return true;
                if (si2.margin < si1.margin) return false;
                // return si1.margin < si2.margin;
                return si1.overlap < si2.overlap;
            }

            SplitInfo()
            {
                // margin = -1;
            }

            SplitInfo(const SplitInfo &r)
            {
                group1 = r.group1;
                group2 = r.group2;
                splitAxis = r.splitAxis;
                sortOrder = r.sortOrder;
                overlap = r.overlap;
                margin = r.margin;
                area = r.area;
                bb1_margin = r.bb1_margin;
                bb2_margin = r.bb2_margin;
                bb1_area = r.bb1_area;
                bb2_area = r.bb2_area;
                increased_area = r.increased_area;
                score = r.score;
                from_rstar = r.from_rstar;

                geometric_properties = r.geometric_properties;

                bb1 = r.bb1;
                bb2 = r.bb2;
                candi_rep_points = r.candi_rep_points;
            }

            SplitInfo(SplitInfo &&r) noexcept
            {
                group1 = std::move(r.group1);
                group2 = std::move(r.group2);
                splitAxis = r.splitAxis;
                sortOrder = r.sortOrder;
                overlap = r.overlap;
                margin = r.margin;
                area = r.area;
                bb1_margin = r.bb1_margin;
                bb2_margin = r.bb2_margin;
                bb1_area = r.bb1_area;
                bb2_area = r.bb2_area;
                increased_area = r.increased_area;
                score = r.score;
                from_rstar = r.from_rstar;

                geometric_properties = std::move(r.geometric_properties);

                bb1 = r.bb1;
                bb2 = r.bb2;
                candi_rep_points = std::move(r.candi_rep_points);
            }

            SplitInfo& operator=(const SplitInfo &r)
            {
                if(this != &r)
                {
                    group1 = r.group1;
                    group2 = r.group2;
                    splitAxis = r.splitAxis;
                    sortOrder = r.sortOrder;
                    overlap = r.overlap;
                    margin = r.margin;
                    area = r.area;
                    bb1_margin = r.bb1_margin;
                    bb2_margin = r.bb2_margin;
                    bb1_area = r.bb1_area;
                    bb2_area = r.bb2_area;
                    increased_area = r.increased_area;
                    score = r.score;
                    from_rstar = r.from_rstar;

                    geometric_properties = r.geometric_properties;

                    bb1 = r.bb1;
                    bb2 = r.bb2;
                    candi_rep_points = r.candi_rep_points;
                }

                return *this;
            }

            SplitInfo& operator=(SplitInfo &&r) noexcept
            {
                group1 = std::move(r.group1);
                group2 = std::move(r.group2);
                splitAxis = r.splitAxis;
                sortOrder = r.sortOrder;
                overlap = r.overlap;
                margin = r.margin;
                area = r.area;
                bb1_margin = r.bb1_margin;
                bb2_margin = r.bb2_margin;
                bb1_area = r.bb1_area;
                bb2_area = r.bb2_area;
                increased_area = r.increased_area;
                score = r.score;
                from_rstar = r.from_rstar;

                geometric_properties = std::move(r.geometric_properties);

                bb1 = r.bb1;
                bb2 = r.bb2;
                candi_rep_points = std::move(r.candi_rep_points);
                return *this;
            }

            void get_candi_rep_points()
            {
                uint64_t dim = bb1.m_dimension;
                uint64_t n_each_dim = 2;
                uint64_t num_points = static_cast<uint64_t>(pow(n_each_dim, dim) + 0.5);
                for (int bb_index = 0; bb_index < 2; ++bb_index)
                {
                    std::vector< std::vector<double> > vecs;
                    if (bb_index == 0)
                    {
                        vecs = {std::vector<double> (bb1.m_pLow, bb1.m_pLow + dim), std::vector<double> (bb1.m_pHigh, bb1.m_pHigh + dim) };
                    }
                    else
                    {
                        vecs = {std::vector<double> (bb2.m_pLow, bb2.m_pLow + dim), std::vector<double> (bb2.m_pHigh, bb2.m_pHigh + dim) };
                    }

                    // Region r = bb1;
                    // if (bb_index == 1)
                    // {
                    //     r = bb2;
                    // }
                    // std::vector<double> low(r.m_pLow, r.m_pLow + dim);
                    // std::vector<double> high(r.m_pHigh, r.m_pHigh + dim);
                    // std::vector< std::vector<double> > vecs = {std::vector<double> (r.m_pLow, r.m_pLow + dim), std::vector<double> (r.m_pHigh, r.m_pHigh + dim) };

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
                        candi_rep_points.emplace_back(std::move(point));
                    }
                }

            }
        };

        class simulateSplitInfo
        {
        public:
            uint32_t splitAxis;
            uint32_t sortOrder;
            double margin;
            double overlap;
            double area;
            double increased_area;
            Region bb1;
            Region bb2;
        };

        class Node : public SpatialIndex::INode
        {
        public:
            // Node(RTree* pTree, Node &ref);
            ~Node() override;

            virtual void cloneFromRef(RTree* refTree, Node& refNode);
            virtual void simpleClone(Node& refNode);

            //
            // Tools::IObject interface
            //
            Tools::IObject* clone() override;

            //
            // Tools::ISerializable interface
            //
            uint32_t getByteArraySize() override;
            void loadFromByteArray(const uint8_t* data) override;
            void storeToByteArray(uint8_t** data, uint32_t& len) override;

            //
            // SpatialIndex::IEntry interface
            //
            id_type getIdentifier() const override;
            void getShape(IShape** out) const override;

            //
            // SpatialIndex::INode interface
            //
            uint32_t getChildrenCount() const override;
            id_type getChildIdentifier(uint32_t index) const override;
            void getChildShape(uint32_t index, IShape** out) const override;
            void getChildData(uint32_t index, uint32_t& length, uint8_t** data) const override;
            uint32_t getLevel() const override;
            bool isIndex() const override;
            bool isLeaf() const override;

            // bool isSplitRecordingTurnedOn() const;
            bool isSplitRecordingTurnedOn() const;
            bool isSubtreeRecordingTurnedOn() const;

        private:
            Node();
            Node(RTree* pTree, id_type id, uint32_t level, uint32_t capacity);


            virtual Node& operator=(const Node&);

            virtual void insertEntry(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id);
            virtual void deleteEntry(uint32_t index);

            virtual bool insertData(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id,
                                    std::stack<id_type>& pathBuffer, uint8_t* overflowTable);
            virtual void reinsertData(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id,
                                      std::vector<uint32_t>& reinsert, std::vector<uint32_t>& keep);

            virtual void rtreeSplit(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id,
                                    std::vector<uint32_t>& group1, std::vector<uint32_t>& group2);
            virtual void rstarSplit(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id,
                                    std::vector<uint32_t>& group1, std::vector<uint32_t>& group2);
            virtual void simulate_rstarSplit(Region& mbr, simulateSplitInfo& ssi);

            virtual uint64_t calc_split_cost_w_queries(std::vector<id_type>& insert_path, std::vector< SplitInfo >& split_infos, uint64_t k, std::vector<Region> &queries, std::vector<uint64_t> &io_cost_each_query);
            virtual void chooseBestSplit(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id,
                                         std::stack<id_type>& pathBuffer, std::vector<uint32_t>& group1,
                                         std::vector<uint32_t>& group2);
            virtual void chooseSplit_w_model(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id,
                                         std::stack<id_type>& pathBuffer, std::vector<uint32_t>& group1,
                                         std::vector<uint32_t>& group2);

            virtual void enumerateRtreeSplit(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id, std::vector<SplitInfo> &split_infos, std::unordered_set< std::string > &existing_split_bbs_str_keys);
            virtual void enumerateRstarSplit(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id, std::vector<SplitInfo>& split_infos, std::unordered_set< std::string > &existing_split_bbs_str_keys);

            virtual void enumerateCandidateSplit(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id, std::vector<SplitInfo>& split_infos);


            virtual void pickSeeds(uint32_t& index1, uint32_t& index2);
            virtual void pickSeeds_w_rv(uint32_t& index1, uint32_t& index2, RTreeVariant rv);

            virtual void condenseTree(std::stack<NodePtr>& toReinsert, std::stack<id_type>& pathBuffer,
                                      NodePtr& ptrThis);

            virtual uint32_t findLeastEnlargement_clone(const Region&) const = 0;
            virtual uint32_t findLeastOverlap_clone(const Region&) const = 0;

            virtual NodePtr chooseSubtree(const Region& mbr, uint32_t level, std::stack<id_type>& pathBuffer) = 0;
            virtual NodePtr findLeaf(const Region& mbr, id_type id, std::stack<id_type>& pathBuffer) = 0;

            virtual void split(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id,
                std::stack<id_type>& pathBuffer, NodePtr& left, NodePtr& right) = 0;

            double getIntersectingAreaBetweenTwoAndOthers(const Region& bb1, const Region &bb2, id_type child_identifier);
            double getIntersectingAreaBetweenOneAndOthers(const Region& bb, id_type child_identifier);

            RTree* m_pTree{nullptr};
            // Parent of all nodes.

            uint32_t m_level{0};
            // The level of the node in the tree.
            // Leaves are always at level 0.

            id_type m_identifier{-1};
            // The unique ID of this node.

            uint32_t m_children{0};
            // The number of children pointed by this node.

            uint32_t m_capacity{0};
            // Specifies the node capacity.

            Region m_nodeMBR;
            // The minimum bounding region enclosing all data contained in the node.

            uint8_t** m_pData{nullptr};
            // The data stored in the node.

            RegionPtr* m_ptrMBR{nullptr};
            // The corresponding data MBRs.

            id_type* m_pIdentifier{nullptr};
            // The corresponding data identifiers.

            uint32_t* m_pDataLength{nullptr};

            uint32_t m_totalDataLength{0};

            class RstarSplitEntry
            {
            public:
                Region* m_pRegion;
                uint32_t m_index;
                uint32_t m_sortDim;

                RstarSplitEntry(Region* pr, uint32_t index, uint32_t dimension) :
                    m_pRegion(pr), m_index(index), m_sortDim(dimension)
                {
                }

                static int compareLow(const void* pv1, const void* pv2)
                {
                    RstarSplitEntry* pe1 = *(RstarSplitEntry**)pv1;
                    RstarSplitEntry* pe2 = *(RstarSplitEntry**)pv2;

                    assert(pe1->m_sortDim == pe2->m_sortDim);

                    if (pe1->m_pRegion->m_pLow[pe1->m_sortDim] < pe2->m_pRegion->m_pLow[pe2->m_sortDim]) return -1;
                    if (pe1->m_pRegion->m_pLow[pe1->m_sortDim] > pe2->m_pRegion->m_pLow[pe2->m_sortDim]) return 1;
                    return 0;
                }

                static int compareHigh(const void* pv1, const void* pv2)
                {
                    RstarSplitEntry* pe1 = *(RstarSplitEntry**)pv1;
                    RstarSplitEntry* pe2 = *(RstarSplitEntry**)pv2;

                    assert(pe1->m_sortDim == pe2->m_sortDim);

                    if (pe1->m_pRegion->m_pHigh[pe1->m_sortDim] < pe2->m_pRegion->m_pHigh[pe2->m_sortDim]) return -1;
                    if (pe1->m_pRegion->m_pHigh[pe1->m_sortDim] > pe2->m_pRegion->m_pHigh[pe2->m_sortDim]) return 1;
                    return 0;
                }
            }; // RstarSplitEntry

            class ReinsertEntry
            {
            public:
                uint32_t m_index;
                double m_dist;

                ReinsertEntry(uint32_t index, double dist) : m_index(index), m_dist(dist)
                {
                }

                static int compareReinsertEntry(const void* pv1, const void* pv2)
                {
                    ReinsertEntry* pe1 = *(ReinsertEntry**)pv1;
                    ReinsertEntry* pe2 = *(ReinsertEntry**)pv2;

                    if (pe1->m_dist < pe2->m_dist) return -1;
                    if (pe1->m_dist > pe2->m_dist) return 1;
                    return 0;
                }
            }; // ReinsertEntry

            // Needed to access protected members without having to cast from Node.
            // It is more efficient than using member functions to access protected members.
            friend class RTree;
            friend class Leaf;
            friend class Index;
            friend class Tools::PointerPool<Node>;
            friend class BulkLoader;
        }; // Node

        int64_t get_split_scores(std::vector< SplitInfo > &split_infos, std::vector< double > &geometric_stats, std::vector< std::vector<uint64_t> > &io_cost_each_split, std::vector< double > &split_scores, bool split_likelihood_as_score, bool print_info=false);
        void get_split_geometric_properties(std::vector< SplitInfo > &split_infos, std::vector< double > &geometric_stats);
        std::string split_bbs_to_string(Region &bb1, Region &bb2);
        void get_padding_split_info(Region &m_nodeMBR, SplitInfo &split_info);


    }
}
#pragma GCC diagnostic pop
