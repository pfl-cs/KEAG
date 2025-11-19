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

namespace SpatialIndex
{
	namespace RTree
	{
		class Index : public Node
		{
		public:
		    // Index(RTree* pTree, Node& ref);
			~Index() override;

		protected:
		    Index(RTree* pTree, id_type id, uint32_t level);

		    NodePtr chooseSubtree(const Region& mbr, uint32_t level, std::stack<id_type>& pathBuffer) override;
			NodePtr findLeaf(const Region& mbr, id_type id, std::stack<id_type>& pathBuffer) override;

		    void split(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id, std::stack<id_type>& pathBuffer, NodePtr& left, NodePtr& right) override;

		    uint32_t findLeastEnlargement_clone(const Region&) const override;
		    uint32_t findLeastOverlap_clone(const Region&) const override;
			uint32_t findLeastEnlargement(const Region&) const;
		    uint32_t findLeastOverlap(const Region&) const;
		    uint32_t findLeastFanout(const Region&) const;
		    uint32_t findBestSubtree(const Region& r, uint32_t insertionLevel) const;
		    uint32_t findSubtree_w_model(const Region& r, uint32_t insertionLevel) const;

		    static uint64_t calcChooseSubtreeCost_w_QueryTree(std::vector<Region> &original_bbs, std::vector< std::vector<Region> > &path_increased_bbs, SubtreeInfo &subtree_info, RTree *query_tree);
		    static uint64_t calcChooseSubtreeCost_w_QueryTree_v2(std::vector<Region> &original_bbs, std::vector< std::vector<Region> > &path_increased_bbs, SubtreeInfo &subtree_info, RTree *query_tree, std::vector<Region> &queries);
		    static uint64_t calcChooseSubtreeCost_w_Queries(std::vector<Region> &original_bbs, std::vector< std::vector<Region> > &path_increased_bbs, std::vector<Region> &queries);

			void adjustTree(Node*, std::stack<id_type>&, bool force = false);
			void adjustTree(Node*, Node*, std::stack<id_type>&, uint8_t* overflowTable);

			class OverlapEntry
			{
			public:
				uint32_t m_index;
			    double m_enlargement;
			    double m_margin_enlargement;
				RegionPtr m_original;
				RegionPtr m_combined;
				double m_oa;
				double m_ca;

			    double m_om;
			    double m_cm;

			    uint32_t m_fanout;

				static int compareEntries(const void* pv1, const void* pv2)
				{
					OverlapEntry* pe1 = * (OverlapEntry**) pv1;
					OverlapEntry* pe2 = * (OverlapEntry**) pv2;

					if (pe1->m_enlargement < pe2->m_enlargement) return -1;
					if (pe1->m_enlargement > pe2->m_enlargement) return 1;
					return 0;
				}
			}; // OverlapEntry

		    void getSubtreeInfo(const Region& r, uint32_t insertionLevel, uint32_t cIndex, OverlapEntry *e, double increased_overlap, std::vector<SubtreeInfo> &subtree_infos) const;
		    void getPaddingSubtreeInfo(const Region& r, std::vector<SubtreeInfo> &subtree_infos) const;
		    void getSubtreeInfo_complex(const Region& r, uint32_t insertionLevel, OverlapEntry *e, double increased_overlap, std::vector<SubtreeInfo> &subtree_infos) const;

			friend class RTree;
			friend class Node;
		    friend class Leaf;
			friend class BulkLoader;
		}; // Index

	    void get_subtree_geometric_properties(const Region &parent_mbr, std::vector< SubtreeInfo > &subtree_infos, std::vector< double > &geometric_stats);
	    int64_t get_subtree_scores(std::vector< SubtreeInfo > &subtree_infos, std::vector< double > &geometric_stats, std::vector< double > &subtree_scores, bool print_info=false);
	    void get_increased_bbs(const Region &original_bb, const Region &combined_bb, std::vector<Region> &increased_bbs);
	}
}
#pragma GCC diagnostic pop
