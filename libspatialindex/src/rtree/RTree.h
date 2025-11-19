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
#include <spatialindex/utils/file_utils.h>
#include "Statistics.h"
#include "Node.h"
#include "Leaf.h"
#include "PointerPoolNode.h"
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>

#include <torch/torch.h>
#include <torch/script.h>
// #include <ATen/Aten.h>

namespace SpatialIndex
{
	namespace RTree
	{
		class RTree : public ISpatialIndex
		{
                  //class NNEntry;

		public:
		    RTree(IStorageManager&, Tools::PropertySet&);
		    RTree(IStorageManager& sm, RTree *refTree, RTreeVariant RV);
				// String                   Value     Description
				// ----------------------------------------------
				// IndexIndentifier         VT_LONG   If specified an existing index will be openened from the supplied
				//                          storage manager with the given index id. Behaviour is unspecified
				//                          if the index id or the storage manager are incorrect.
				// Dimension                VT_ULONG  Dimensionality of the data that will be inserted.
				// IndexCapacity            VT_ULONG  The index node capacity. Default is 100.
				// LeafCapactiy             VT_ULONG  The leaf node capacity. Default is 100.
				// FillFactor               VT_DOUBLE The fill factor. Default is 70%
				// TreeVariant              VT_LONG   Can be one of Linear, Quadratic or Rstar. Default is Rstar
				// NearMinimumOverlapFactor VT_ULONG  Default is 32.
				// SplitDistributionFactor  VT_DOUBLE Default is 0.4
				// ReinsertFactor           VT_DOUBLE Default is 0.3
				// EnsureTightMBRs          VT_BOOL   Default is true
				// IndexPoolCapacity        VT_LONG   Default is 100
				// LeafPoolCapacity         VT_LONG   Default is 100
				// RegionPoolCapacity       VT_LONG   Default is 1000
				// PointPoolCapacity        VT_LONG   Default is 500

			~RTree() ;

			//
			// ISpatialIndex interface
			//
		    void insertData(uint32_t len, const uint8_t* pData, const IShape& shape, id_type shapeIdentifier) override;
			bool deleteData(const IShape& shape, id_type id) override;
			void internalNodesQuery(const IShape& query, IVisitor& v) override;
			void containsWhatQuery(const IShape& query, IVisitor& v) override;
		    void intersectsWithQuery(const IShape& query, IVisitor& v) override;
		    uint64_t intersectsWithQueryCount(const IShape& query) override;
		    uint64_t getIntersectsWithQueryIOCost(const IShape& query) override;
			void pointLocationQuery(const Point& query, IVisitor& v) override;
			void nearestNeighborQuery(uint32_t k, const IShape& query, IVisitor& v, INearestNeighborComparator&) override;
			void nearestNeighborQuery(uint32_t k, const IShape& query, IVisitor& v) override;
		    uint64_t getKNNQueryIOCost(uint32_t k, const IShape& query) override;

            void selfJoinQuery(const IShape& s, IVisitor& v) override;
			void queryStrategy(IQueryStrategy& qs) override;
			void getIndexProperties(Tools::PropertySet& out) const override;
			void addCommand(ICommand* pCommand, CommandType ct) override;
			bool isIndexValid() override;
			void getStatistics(IStatistics** out) const override;
            void flush() override;

		    inline uint32_t getRootLevel() override { return readNode(m_rootID)->getLevel(); }
		    inline void setQueryTree(ISpatialIndex* _queryTree) override { this->queryTree = reinterpret_cast<RTree*>(_queryTree); }
		    inline void setLbds(std::vector<double> &_lbds) override { lbds = _lbds; }
		    inline void setUbds(std::vector<double> &_ubds) override { ubds = _ubds; }
		    void setDeviceConfigs() override;

		    // Split-related functions
		    inline void turnOnSplitRecording() override { split_recording = true; }
		    inline void turnOffSplitRecording() override { split_recording = false; }
		    inline void setAppendSplits(bool _append_splits) override { append_splits = _append_splits; }
		    void initSplitRecording(const std::string &path) override;
		    void stopSplitRecording() override { assert(split_recording == false); split_records_file.close(); split_records_text_file.close(); }
		    void writeASplitRecord(Region &m_nodeMBR, std::vector<Region> &child_mbrs, std::vector<SplitInfo>& splitInfos, std::vector<double> &split_scores, bool isLeaf);
		    
		    inline void setSplitDebug() override { split_debug = true; }
		    inline uint64_t getNumSplits() override { return m_stats.getSplits(); }
		    inline void setSplitNormalizationInfosPath(std::string &_split_normalization_infos_path) override { split_normalization_infos_path = _split_normalization_infos_path;}
		    inline void setSplitModelPath(std::string &_split_model_path) override {split_model_path = _split_model_path;}
		    void loadSplitModel() override;
		    void loadSplitNormalizationInfos() override;
		    int64_t splitPredict(Region& m_nodeMBR, std::vector<Region>& child_mbrs, std::vector<SplitInfo>& split_infos, bool isLeaf);

		    // ChooseSubtree-related functions
		    inline void turnOnSubtreeRecording() override { subtree_recording = true; }
		    inline void turnOffSubtreeRecording() override { subtree_recording = false; }
		    inline void setAppendSubtrees(bool _append_subtrees) override { append_subtrees = _append_subtrees; }
		    void initSubtreeRecording(const std::string &path) override;
		    void stopSubtreeRecording() override { assert(subtree_recording == false); subtree_records_file.close(); subtree_records_text_file.close(); }
		    void writeASubtreeRecord(const Region &object, const Region &m_nodeMBR, std::vector<SubtreeInfo>& subtree_infos, std::vector<double> &subtree_scores, uint32_t level);

		    inline void setSubtreeDebug() override { subtree_debug = true; }
		    inline void setSubtreeNormalizationInfosPath(std::string &_subtree_normalization_infos_path) override { subtree_normalization_infos_path = _subtree_normalization_infos_path;}
		    inline void setSubtreeModelPath(std::string &_subtree_model_path) override {subtree_model_path = _subtree_model_path;}
		    void loadSubtreeModel() override;
		    void loadSubtreeNormalizationInfos() override;
		    int64_t subtreePredict(const Region &object, const Region& m_nodeMBR, std::vector<SubtreeInfo>& subtree_infos,  uint32_t level);

		private:
			void initNew(Tools::PropertySet&);
			void initOld(Tools::PropertySet& ps);
			void storeHeader();
			void loadHeader();

			void insertData_impl(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id);
			void insertData_impl(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id, uint32_t level, uint8_t* overflowTable);
			bool deleteData_impl(const Region& mbr, id_type id);

		    id_type writeNode(Node*);
		    id_type writeNode_wo_modify_stats(Node*);
		    NodePtr readNode(id_type page);
		    NodePtr readNode_wo_modify_stats(id_type page);
			void deleteNode(Node*);

		    void rangeQuery(RangeQueryType type, const IShape& query, IVisitor& v);
		    uint64_t rangeQueryCount(RangeQueryType type, const IShape& query);
		    uint64_t rangeQueryIOCost(RangeQueryType type, const IShape& query);
		    void getOverlapMBRs(const IShape& query, std::vector<Region>& mbrs) override;
		    void getRestrictedOverlapMBRIDs(const Region& query, const Region &ignore, std::unordered_set<id_type>& mbr_ids);
		    void getRestrictedOverlapNodeIDs(const Region& query, const Region &ignore, std::unordered_set<id_type>& node_ids);
		    uint64_t countRestrictedOverlapMBRs(const Region& query, const Region &ignore);
            void selfJoinQuery(id_type id1, id_type id2, const Region& r, IVisitor& vis);
			void visitSubTree(NodePtr subTree, IVisitor& v);

			IStorageManager* m_pStorageManager;

			id_type m_rootID, m_headerID;

			RTreeVariant m_treeVariant;

			double m_fillFactor;

			uint32_t m_indexCapacity;

			uint32_t m_leafCapacity;

			uint32_t m_nearMinimumOverlapFactor;
				// The R*-Tree 'p' constant, for calculating nearly minimum overlap cost.
				// [Beckmann, Kriegel, Schneider, Seeger 'The R*-tree: An efficient and Robust Access Method
				// for Points and Rectangles', Section 4.1]

			double m_splitDistributionFactor;
				// The R*-Tree 'm' constant, for calculating spliting distributions.
				// [Beckmann, Kriegel, Schneider, Seeger 'The R*-tree: An efficient and Robust Access Method
				// for Points and Rectangles', Section 4.2]

			double m_reinsertFactor;
				// The R*-Tree 'p' constant, for removing entries at reinserts.
				// [Beckmann, Kriegel, Schneider, Seeger 'The R*-tree: An efficient and Robust Access Method
				//  for Points and Rectangles', Section 4.3]

			uint32_t m_dimension;

			Region m_infiniteRegion;

			Statistics m_stats;

			bool m_bTightMBRs;

			Tools::PointerPool<Point> m_pointPool;
			Tools::PointerPool<Region> m_regionPool;
			Tools::PointerPool<Node> m_indexPool;
			Tools::PointerPool<Node> m_leafPool;

			std::vector<std::shared_ptr<ICommand> > m_writeNodeCommands;
			std::vector<std::shared_ptr<ICommand> > m_readNodeCommands;
			std::vector<std::shared_ptr<ICommand> > m_deleteNodeCommands;

		    //--------------------Global information-----------------------
		    std::vector<double> lbds;
		    std::vector<double> ubds;

		    RTree *queryTree;

		    torch::TensorOptions options;
		    torch::TensorOptions cpu_options;
		    std::string device = "cpu";

		    bool cuda_exists = false;

		    //--------------------Used for split -----------------------
		    bool split_recording = false;
		    std::ofstream split_records_file; //(data_path, std::ios_base::out | std::ios_base::binary);
		    std::ofstream split_records_text_file; //(data_path, std::ios_base::out | std::ios_base::binary);
		    bool append_splits = false;
		    bool split_likelihood_as_score = false;
		    bool split_debug = false;

		    struct splitModelConfig
		    {
		        std::vector<int64_t> split_context_rep_points_shape;
		        std::vector<int64_t> split_candi_rep_points_shape;
		        std::vector<int64_t> split_geometric_properties_shape;

		        int64_t max_num_splits = -1;
		        int64_t num_maps_each_point = -1;
		        int64_t num_split_context_rep_points = -1;
		        int64_t num_candi_rep_points_per_split = -1;
		        int64_t num_geometric_properties = -1;

		        std::vector<double> points_mean;
		        std::vector<double> points_std;
		        std::vector<double> geometric_properties_mean;
		        std::vector<double> geometric_properties_std;

		    } split_model_config_;

		    std::vector<float> input_split_context_rep_points;
		    std::vector<float> input_split_candi_rep_points;
		    std::vector<float> input_split_geometric_properties;
		    std::string split_normalization_infos_path;
		    std::string split_model_path;
		    torch::jit::script::Module split_model;

		    //--------------------Used for subtree -----------------------
		    bool subtree_recording = false;
		    std::ofstream subtree_records_file; //(data_path, std::ios_base::out | std::ios_base::binary);
		    std::ofstream subtree_records_text_file; //(data_path, std::ios_base::out | std::ios_base::binary);
		    bool append_subtrees = false;
		    bool subtree_debug = false;

		    uint64_t num_find_best_subtree = 0;

		    struct subtreeModelConfig
		    {
		        std::vector<int64_t> subtree_context_rep_points_shape;
		        std::vector<int64_t> subtree_candi_rep_points_shape;
		        std::vector<int64_t> subtree_geometric_properties_shape;

		        int64_t max_num_subtrees = -1;
		        int64_t num_maps_each_point = -1;
		        int64_t num_subtree_context_rep_points = -1;
		        int64_t num_candi_rep_points_per_subtree = -1;
		        int64_t num_geometric_properties = -1;

		        std::vector<double> points_mean;
		        std::vector<double> points_std;
		        std::vector<double> geometric_properties_mean;
		        std::vector<double> geometric_properties_std;

		    } subtree_model_config_;

		    std::vector<float> input_subtree_context_rep_points;
		    std::vector<float> input_subtree_candi_rep_points;
		    std::vector<float> input_subtree_geometric_properties;
		    std::string subtree_normalization_infos_path;
		    std::string subtree_model_path;
		    torch::jit::script::Module subtree_model;

		    int64_t num_subtrees_threshold = 10;


		    //--------------------Used for the reference tree-----------------------
		    RTree *refTree;

		    //-------------------------------end------------------------------------
			class NNEntry
			{
			public:
				id_type m_id;
				IEntry* m_pEntry;
				double m_minDist;

				NNEntry(id_type id, IEntry* e, double f) : m_id(id), m_pEntry(e), m_minDist(f) {}
				~NNEntry() = default;

			}; // NNEntry

			class NNComparator : public INearestNeighborComparator
			{
			public:
				double getMinimumDistance(const IShape& query, const IShape& entry) 
				{
					return query.getMinimumDistance(entry);
				}

				double getMinimumDistance(const IShape& query, const IData& data) 
				{
					IShape* pS;
					data.getShape(&pS);
					double ret = query.getMinimumDistance(*pS);
					delete pS;
					return ret;
				}
			}; // NNComparator

			class ValidateEntry
			{
			public:
				ValidateEntry(Region& r, NodePtr& pNode) : m_parentMBR(r), m_pNode(pNode) {}

				Region m_parentMBR;
				NodePtr m_pNode;
			}; // ValidateEntry

			friend class Node;
			friend class Leaf;
			friend class Index;
			friend class BulkLoader;

			friend std::ostream& operator<<(std::ostream& os, const RTree& t);
		}; // RTree

		std::ostream& operator<<(std::ostream& os, const RTree& t);
	    void get_split_rep_points(const Region &r, std::vector< std::vector<double> > &res);
	    void get_region_boundary_points(const Region &r, std::vector< std::vector<double> > &res);
	}
}
