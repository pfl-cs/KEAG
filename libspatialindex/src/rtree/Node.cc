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
#include <unordered_set>
#include <iostream>
#include <string>
#include <format>
// #include <cassert>

#include <spatialindex/SpatialIndex.h>

#include "RTree.h"
#include "Node.h"
#include "Index.h"
#include "Leaf.h"

using namespace SpatialIndex;
using namespace SpatialIndex::RTree;

//
// Tools::IObject interface
//
Tools::IObject* Node::clone()
{
	throw Tools::NotSupportedException("IObject::clone should never be called.");
}

//
// Tools::ISerializable interface
//
uint32_t Node::getByteArraySize()
{
	return
		(sizeof(uint32_t) +
		sizeof(uint32_t) +
		sizeof(uint32_t) +
		(m_children * (m_pTree->m_dimension * sizeof(double) * 2 + sizeof(id_type) + sizeof(uint32_t))) +
		m_totalDataLength +
		(2 * m_pTree->m_dimension * sizeof(double)));
}

void Node::loadFromByteArray(const uint8_t* ptr)
{
	m_nodeMBR = m_pTree->m_infiniteRegion;

	// skip the node type information, it is not needed.
	ptr += sizeof(uint32_t);

	memcpy(&m_level, ptr, sizeof(uint32_t));
	ptr += sizeof(uint32_t);

	memcpy(&m_children, ptr, sizeof(uint32_t));
	ptr += sizeof(uint32_t);

	for (uint32_t u32Child = 0; u32Child < m_children; ++u32Child)
	{
		m_ptrMBR[u32Child] = m_pTree->m_regionPool.acquire();
		*(m_ptrMBR[u32Child]) = m_pTree->m_infiniteRegion;

		memcpy(m_ptrMBR[u32Child]->m_pLow, ptr, m_pTree->m_dimension * sizeof(double));
		ptr += m_pTree->m_dimension * sizeof(double);
		memcpy(m_ptrMBR[u32Child]->m_pHigh, ptr, m_pTree->m_dimension * sizeof(double));
		ptr += m_pTree->m_dimension * sizeof(double);
		memcpy(&(m_pIdentifier[u32Child]), ptr, sizeof(id_type));
		ptr += sizeof(id_type);

		memcpy(&(m_pDataLength[u32Child]), ptr, sizeof(uint32_t));
		ptr += sizeof(uint32_t);

		if (m_pDataLength[u32Child] > 0)
		{
			m_totalDataLength += m_pDataLength[u32Child];
			m_pData[u32Child] = new uint8_t[m_pDataLength[u32Child]];
			memcpy(m_pData[u32Child], ptr, m_pDataLength[u32Child]);
			ptr += m_pDataLength[u32Child];
		}
		else
		{
			m_pData[u32Child] = nullptr;
		}

		//m_nodeMBR.combineRegion(*(m_ptrMBR[u32Child]));
	}

	memcpy(m_nodeMBR.m_pLow, ptr, m_pTree->m_dimension * sizeof(double));
	ptr += m_pTree->m_dimension * sizeof(double);
	memcpy(m_nodeMBR.m_pHigh, ptr, m_pTree->m_dimension * sizeof(double));
	//ptr += m_pTree->m_dimension * sizeof(double);
}

void Node::storeToByteArray(uint8_t** data, uint32_t& len)
{
	len = getByteArraySize();

	*data = new uint8_t[len];
	uint8_t* ptr = *data;

	uint32_t nodeType;

	if (m_level == 0) nodeType = PersistentLeaf;
	else nodeType = PersistentIndex;

	memcpy(ptr, &nodeType, sizeof(uint32_t));
	ptr += sizeof(uint32_t);

	memcpy(ptr, &m_level, sizeof(uint32_t));
	ptr += sizeof(uint32_t);

	memcpy(ptr, &m_children, sizeof(uint32_t));
	ptr += sizeof(uint32_t);

	for (uint32_t u32Child = 0; u32Child < m_children; ++u32Child)
	{
		memcpy(ptr, m_ptrMBR[u32Child]->m_pLow, m_pTree->m_dimension * sizeof(double));
		ptr += m_pTree->m_dimension * sizeof(double);
		memcpy(ptr, m_ptrMBR[u32Child]->m_pHigh, m_pTree->m_dimension * sizeof(double));
		ptr += m_pTree->m_dimension * sizeof(double);
		memcpy(ptr, &(m_pIdentifier[u32Child]), sizeof(id_type));
		ptr += sizeof(id_type);

		memcpy(ptr, &(m_pDataLength[u32Child]), sizeof(uint32_t));
		ptr += sizeof(uint32_t);

		if (m_pDataLength[u32Child] > 0)
		{
			memcpy(ptr, m_pData[u32Child], m_pDataLength[u32Child]);
			ptr += m_pDataLength[u32Child];
		}
	}

	// store the node MBR for efficiency. This increases the node size a little bit.
	memcpy(ptr, m_nodeMBR.m_pLow, m_pTree->m_dimension * sizeof(double));
	ptr += m_pTree->m_dimension * sizeof(double);
	memcpy(ptr, m_nodeMBR.m_pHigh, m_pTree->m_dimension * sizeof(double));
	//ptr += m_pTree->m_dimension * sizeof(double);

	assert(len == (ptr - *data) + m_pTree->m_dimension * sizeof(double));
}

//
// SpatialIndex::IEntry interface
//
SpatialIndex::id_type Node::getIdentifier() const
{
	return m_identifier;
}

void Node::getShape(IShape** out) const
{
	*out = new Region(m_nodeMBR);
}

//
// SpatialIndex::INode interface
//
uint32_t Node::getChildrenCount() const
{
	return m_children;
}

SpatialIndex::id_type Node::getChildIdentifier(uint32_t index) const
{
	if (index >= m_children) throw Tools::IndexOutOfBoundsException(index);

	return m_pIdentifier[index];
}

void Node::getChildShape(uint32_t index, IShape** out) const
{
	if (index >= m_children) throw Tools::IndexOutOfBoundsException(index);

	*out = new Region(*(m_ptrMBR[index]));
}

void Node::getChildData(uint32_t index, uint32_t& length, uint8_t** data) const
{
	if (index >= m_children) throw Tools::IndexOutOfBoundsException(index);
	if (m_pData[index] == nullptr)
	{
		length = 0;
		data = nullptr;
	}
	else
	{
		length = m_pDataLength[index];
		*data = m_pData[index];
	}
}

uint32_t Node::getLevel() const
{
	return m_level;
}

bool Node::isLeaf() const
{
	return (m_level == 0);
}

bool Node::isIndex() const
{
	return (m_level != 0);
}

bool Node::isSplitRecordingTurnedOn() const
{
    return m_pTree->split_recording;
}

bool Node::isSubtreeRecordingTurnedOn() const
{
    return m_pTree->subtree_recording;
}

//
// Internal
//

Node::Node()
= default;

Node::Node(SpatialIndex::RTree::RTree* pTree, id_type id, uint32_t level, uint32_t capacity) :
	m_pTree(pTree),
	m_level(level),
	m_identifier(id),
	m_children(0),
	m_capacity(capacity),
	m_pData(nullptr),
	m_ptrMBR(nullptr),
	m_pIdentifier(nullptr),
	m_pDataLength(nullptr),
	m_totalDataLength(0)
{
	m_nodeMBR.makeInfinite(m_pTree->m_dimension);

	try
	{
		m_pDataLength = new uint32_t[m_capacity + 1];
		m_pData = new uint8_t*[m_capacity + 1];
		m_ptrMBR = new RegionPtr[m_capacity + 1];
		m_pIdentifier = new id_type[m_capacity + 1];
	}
	catch (...)
	{
		delete[] m_pDataLength;
		delete[] m_pData;
		delete[] m_ptrMBR;
		delete[] m_pIdentifier;
		throw;
	}
}

void Node::cloneFromRef(RTree *refTree, Node &refNode)
{
    m_level = refNode.m_level;
    m_children = refNode.m_children;
    m_capacity = refNode.m_capacity;
    m_nodeMBR = refNode.m_nodeMBR;

    m_totalDataLength = refNode.m_totalDataLength;
    m_pDataLength = new uint32_t[m_capacity + 1];
    m_pData = new uint8_t*[m_capacity + 1];
    m_ptrMBR = new RegionPtr[m_capacity + 1];
    m_pIdentifier = new id_type[m_capacity + 1];

    memcpy(m_pDataLength, refNode.m_pDataLength, m_capacity * sizeof(uint32_t));
    // memcpy(m_pIdentifier, ref.m_pIdentifier, m_capacity * sizeof(id_type));

    for (uint32_t u32Child = 0; u32Child < m_children; ++u32Child)
    {
        m_ptrMBR[u32Child] = m_pTree->m_regionPool.acquire();
        *(m_ptrMBR[u32Child]) = m_pTree->m_infiniteRegion;


        if (m_ptrMBR[u32Child]->m_dimension != refNode.m_pTree->m_dimension)
        {
            int left = m_ptrMBR[u32Child]->m_dimension;
            int right = refNode.m_pTree->m_dimension;
            throw Tools::IsNotEqualException<int>("Node::cloneFromRef", "m_ptrMBR[u32Child]->m_dimension", "m_pTree->m_dimension", left, right);
        }
        memcpy(m_ptrMBR[u32Child]->m_pLow, (refNode.m_ptrMBR)[u32Child]->m_pLow, m_pTree->m_dimension * sizeof(double));
        memcpy(m_ptrMBR[u32Child]->m_pHigh, (refNode.m_ptrMBR)[u32Child]->m_pHigh, m_pTree->m_dimension * sizeof(double));

        if (m_pDataLength[u32Child] > 0)
        {
            m_pData[u32Child] = new uint8_t[m_pDataLength[u32Child]];
            memcpy(m_pData[u32Child], (refNode.m_pData)[u32Child], m_pDataLength[u32Child]);
        }
        else
        {
            m_pData[u32Child] = nullptr;
        }

        if (m_level > 1)
        {
            NodePtr childRefNode = refTree->readNode_wo_modify_stats(refNode.m_pIdentifier[u32Child]);
            Index childNode(m_pTree, -1, m_level - 1);
            // childNode.cloneFromRef(refTree, childRefNode);
            childNode.cloneFromRef(refTree, *childRefNode);
            m_pIdentifier[u32Child] = m_pTree->writeNode_wo_modify_stats(&childNode);
        }
        else if (m_level == 1)
        {
            NodePtr childRefNode = refTree->readNode_wo_modify_stats(refNode.m_pIdentifier[u32Child]);
            Leaf childNode(m_pTree, -1);
            childNode.cloneFromRef(refTree, *childRefNode);
            m_pIdentifier[u32Child] = m_pTree->writeNode_wo_modify_stats(&childNode);
        }
    }
}

void Node::simpleClone(Node &refNode)
{
    m_level = refNode.m_level;
    m_children = refNode.m_children;
    m_capacity = refNode.m_capacity;
    m_nodeMBR = refNode.m_nodeMBR;

    m_totalDataLength = refNode.m_totalDataLength;
    m_pDataLength = new uint32_t[m_capacity + 1];
    // m_pData = new uint8_t*[m_capacity + 1];
    m_pData = nullptr;
    m_ptrMBR = new RegionPtr[m_capacity + 1];
    m_pIdentifier = new id_type[m_capacity + 1];

    // memcpy(m_pDataLength, refNode.m_pDataLength, m_capacity * sizeof(uint32_t));
    memcpy(m_pIdentifier, refNode.m_pIdentifier, m_capacity * sizeof(id_type));

    for (uint32_t u32Child = 0; u32Child < m_children; ++u32Child)
    {
        m_ptrMBR[u32Child] = m_pTree->m_regionPool.acquire();
        *(m_ptrMBR[u32Child]) = m_pTree->m_infiniteRegion;

        if (m_ptrMBR[u32Child]->m_dimension != refNode.m_pTree->m_dimension)
        {
            int left = m_ptrMBR[u32Child]->m_dimension;
            int right = refNode.m_pTree->m_dimension;
            throw Tools::IsNotEqualException<int>("Node::simpleClone", "m_ptrMBR[u32Child]->m_dimension", "m_pTree->m_dimension", left, right);
        }

        memcpy(m_ptrMBR[u32Child]->m_pLow, (refNode.m_ptrMBR)[u32Child]->m_pLow, m_pTree->m_dimension * sizeof(double));
        memcpy(m_ptrMBR[u32Child]->m_pHigh, (refNode.m_ptrMBR)[u32Child]->m_pHigh, m_pTree->m_dimension * sizeof(double));
    }
}


Node::~Node()
{
	if (m_pData != nullptr)
	{
		for (uint32_t u32Child = 0; u32Child < m_children; ++u32Child)
		{
			if (m_pData[u32Child] != nullptr) delete[] m_pData[u32Child];
		}

		delete[] m_pData;
	}

	delete[] m_pDataLength;
	delete[] m_ptrMBR;
	delete[] m_pIdentifier;
}

Node& Node::operator=(const Node&)
{
	throw Tools::IllegalStateException("operator =: This should never be called.");
}

void Node::insertEntry(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id)
{
	assert(m_children < m_capacity);

	m_pDataLength[m_children] = dataLength;
	m_pData[m_children] = pData;
	m_ptrMBR[m_children] = m_pTree->m_regionPool.acquire();
	*(m_ptrMBR[m_children]) = mbr;
	m_pIdentifier[m_children] = id;

	m_totalDataLength += dataLength;
	++m_children;

	m_nodeMBR.combineRegion(mbr);
}

void Node::deleteEntry(uint32_t index)
{
	assert(index >= 0 && index < m_children);

	// cache it, since I might need it for "touches" later.
	RegionPtr ptrR = m_ptrMBR[index];

	m_totalDataLength -= m_pDataLength[index];
	if (m_pData[index] != nullptr) delete[] m_pData[index];

	if (m_children > 1 && index != m_children - 1)
	{
		m_pDataLength[index] = m_pDataLength[m_children - 1];
		m_pData[index] = m_pData[m_children - 1];
		m_ptrMBR[index] = m_ptrMBR[m_children - 1];
		m_pIdentifier[index] = m_pIdentifier[m_children - 1];
	}

	--m_children;

	// WARNING: index has now changed. Do not use it below here.

	if (m_children == 0)
	{
		m_nodeMBR = m_pTree->m_infiniteRegion;
	}
	else if (m_pTree->m_bTightMBRs && m_nodeMBR.touchesRegion(*ptrR))
	{
		for (uint32_t cDim = 0; cDim < m_nodeMBR.m_dimension; ++cDim)
		{
			m_nodeMBR.m_pLow[cDim] = std::numeric_limits<double>::max();
			m_nodeMBR.m_pHigh[cDim] = -std::numeric_limits<double>::max();

			for (uint32_t u32Child = 0; u32Child < m_children; ++u32Child)
			{
				m_nodeMBR.m_pLow[cDim] = std::min(m_nodeMBR.m_pLow[cDim], m_ptrMBR[u32Child]->m_pLow[cDim]);
				m_nodeMBR.m_pHigh[cDim] = std::max(m_nodeMBR.m_pHigh[cDim], m_ptrMBR[u32Child]->m_pHigh[cDim]);
			}
		}
	}
}

bool Node::insertData(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id, std::stack<id_type>& pathBuffer, uint8_t* overflowTable)
{
	if (m_children < m_capacity)
	{
		bool adjusted = false;

		// this has to happen before insertEntry modifies m_nodeMBR.
		bool b = m_nodeMBR.containsRegion(mbr);

		insertEntry(dataLength, pData, mbr, id);
		m_pTree->writeNode(this);

		if ((! b) && (! pathBuffer.empty()))
		{
			id_type cParent = pathBuffer.top(); pathBuffer.pop();
			NodePtr ptrN = m_pTree->readNode(cParent);
			Index* p = static_cast<Index*>(ptrN.get());
			p->adjustTree(this, pathBuffer);
			adjusted = true;
		}

		return adjusted;
	}
	// else if (m_pTree->m_treeVariant == RV_RSTAR && (! pathBuffer.empty()) && overflowTable[m_level] == 0)
	    else if (m_pTree->m_treeVariant == RV_RSTAR && (! pathBuffer.empty()) && overflowTable[m_level] == 2)
	// else if ((m_pTree->m_treeVariant == RV_RSTAR || m_pTree->m_treeVariant == RV_BEST_SPLIT || m_pTree->m_treeVariant == RV_MODEL_SPLIT) && (! pathBuffer.empty()) && overflowTable[m_level] == 0
	// && (! pathBuffer.empty()) && overflowTable[m_level] == 0)
	{
	        throw Tools::BasicException("In Node::insertData, this function should not be called.");
		overflowTable[m_level] = 1;

		std::vector<uint32_t> vReinsert, vKeep;
		reinsertData(dataLength, pData, mbr, id, vReinsert, vKeep);

		uint32_t lReinsert = static_cast<uint32_t>(vReinsert.size());
		uint32_t lKeep = static_cast<uint32_t>(vKeep.size());

		uint8_t** reinsertdata = nullptr;
		RegionPtr* reinsertmbr = nullptr;
		id_type* reinsertid = nullptr;
		uint32_t* reinsertlen = nullptr;
		uint8_t** keepdata = nullptr;
		RegionPtr* keepmbr = nullptr;
		id_type* keepid = nullptr;
		uint32_t* keeplen = nullptr;

		try
		{
			reinsertdata = new uint8_t*[lReinsert];
			reinsertmbr = new RegionPtr[lReinsert];
			reinsertid = new id_type[lReinsert];
			reinsertlen = new uint32_t[lReinsert];

			keepdata = new uint8_t*[m_capacity + 1];
			keepmbr = new RegionPtr[m_capacity + 1];
			keepid = new id_type[m_capacity + 1];
			keeplen = new uint32_t[m_capacity + 1];
		}
		catch (...)
		{
			delete[] reinsertdata;
			delete[] reinsertmbr;
			delete[] reinsertid;
			delete[] reinsertlen;
			delete[] keepdata;
			delete[] keepmbr;
			delete[] keepid;
			delete[] keeplen;
			throw;
		}

		uint32_t cIndex;

		for (cIndex = 0; cIndex < lReinsert; ++cIndex)
		{
			reinsertlen[cIndex] = m_pDataLength[vReinsert[cIndex]];
			reinsertdata[cIndex] = m_pData[vReinsert[cIndex]];
			reinsertmbr[cIndex] = m_ptrMBR[vReinsert[cIndex]];
			reinsertid[cIndex] = m_pIdentifier[vReinsert[cIndex]];
		}

		for (cIndex = 0; cIndex < lKeep; ++cIndex)
		{
			keeplen[cIndex] = m_pDataLength[vKeep[cIndex]];
			keepdata[cIndex] = m_pData[vKeep[cIndex]];
			keepmbr[cIndex] = m_ptrMBR[vKeep[cIndex]];
			keepid[cIndex] = m_pIdentifier[vKeep[cIndex]];
		}

		delete[] m_pDataLength;
		delete[] m_pData;
		delete[] m_ptrMBR;
		delete[] m_pIdentifier;

		m_pDataLength = keeplen;
		m_pData = keepdata;
		m_ptrMBR = keepmbr;
		m_pIdentifier = keepid;
		m_children = lKeep;
		m_totalDataLength = 0;

		for (uint32_t u32Child = 0; u32Child < m_children; ++u32Child) m_totalDataLength += m_pDataLength[u32Child];

		for (uint32_t cDim = 0; cDim < m_nodeMBR.m_dimension; ++cDim)
		{
			m_nodeMBR.m_pLow[cDim] = std::numeric_limits<double>::max();
			m_nodeMBR.m_pHigh[cDim] = -std::numeric_limits<double>::max();

			for (uint32_t u32Child = 0; u32Child < m_children; ++u32Child)
			{
				m_nodeMBR.m_pLow[cDim] = std::min(m_nodeMBR.m_pLow[cDim], m_ptrMBR[u32Child]->m_pLow[cDim]);
				m_nodeMBR.m_pHigh[cDim] = std::max(m_nodeMBR.m_pHigh[cDim], m_ptrMBR[u32Child]->m_pHigh[cDim]);
			}
		}

		m_pTree->writeNode(this);

		// Divertion from R*-Tree algorithm here. First adjust
		// the path to the root, then start reinserts, to avoid complicated handling
		// of changes to the same node from multiple insertions.
		id_type cParent = pathBuffer.top(); pathBuffer.pop();
		NodePtr ptrN = m_pTree->readNode(cParent);
		Index* p = static_cast<Index*>(ptrN.get());
		p->adjustTree(this, pathBuffer, true);

		for (cIndex = 0; cIndex < lReinsert; ++cIndex)
		{
			m_pTree->insertData_impl(
				reinsertlen[cIndex], reinsertdata[cIndex],
				*(reinsertmbr[cIndex]), reinsertid[cIndex],
				m_level, overflowTable);
		}

		delete[] reinsertdata;
		delete[] reinsertmbr;
		delete[] reinsertid;
		delete[] reinsertlen;

		return true;
	}
	else
	{
		NodePtr n;
		NodePtr nn;
	    // if (m_level != -1)
	    // {
	    //     std::stack<id_type> path_backup(pathBuffer);
	    //     std::cout << "root.level = " << m_pTree->getRootLevel() << " " << ", m_level = " << m_level << ", path info: ";
	    //     while (path_backup.size() > 0)
	    //     {
	    //         id_type cParent = path_backup.top(); path_backup.pop();
	    //         NodePtr p = m_pTree->readNode(cParent);
	    //         std::cout << "(" << cParent << ", " << p->getLevel() << ") ";
	    //     }
	    //     std::cout << std::endl;
	    // }

	    // if (isSplitRecordingTurnedOn())
	    // {
	    uint64_t l1 = pathBuffer.size();
	    split(dataLength, pData, mbr, id, pathBuffer, n, nn);
	    uint64_t l2 = pathBuffer.size();
	    if (l1 != l2)
	    {
	        throw Tools::IsNotEqualException<uint64_t>("Node::insertData", "l1", "l2", l1, l2);
	    }
	    // }
	    // else
	    // {
	    //     split(dataLength, pData, mbr, id, pathBuffer, n, nn);
	    // }
	    // if (m_level != -1)
	    // {
	    //     std::stack<id_type> path_backup(pathBuffer);
	    //     std::cout << "root.level = " << m_pTree->getRootLevel() << " " << ", m_level = " << m_level << ", path info: ";
	    //     while (path_backup.size() > 0)
	    //     {
	    //         id_type cParent = path_backup.top(); path_backup.pop();
	    //         NodePtr p = m_pTree->readNode(cParent);
	    //         std::cout << "(" << cParent << ", " << p->getLevel() << ") ";
	    //     }
	    //     std::cout << std::endl;
	    // }

		if (pathBuffer.empty())
		{
			n->m_level = m_level;
			nn->m_level = m_level;
			n->m_identifier = -1;
			nn->m_identifier = -1;
			m_pTree->writeNode(n.get());
			m_pTree->writeNode(nn.get());

			NodePtr ptrR = m_pTree->m_indexPool.acquire();
			if (ptrR.get() == nullptr)
			{
				ptrR = NodePtr(new Index(m_pTree, m_pTree->m_rootID, m_level + 1), &(m_pTree->m_indexPool));
			}
			else
			{
				//ptrR->m_pTree = m_pTree;
				ptrR->m_identifier = m_pTree->m_rootID;
				ptrR->m_level = m_level + 1;
				ptrR->m_nodeMBR = m_pTree->m_infiniteRegion;
			}

			ptrR->insertEntry(0, nullptr, n->m_nodeMBR, n->m_identifier);
			ptrR->insertEntry(0, nullptr, nn->m_nodeMBR, nn->m_identifier);

			m_pTree->writeNode(ptrR.get());

			m_pTree->m_stats.m_nodesInLevel[m_level] = 2;
			m_pTree->m_stats.m_nodesInLevel.push_back(1);
			m_pTree->m_stats.m_u32TreeHeight = m_level + 2;
		}
		else
		{
			n->m_level = m_level;
			nn->m_level = m_level;
			n->m_identifier = m_identifier;
			nn->m_identifier = -1;

			m_pTree->writeNode(n.get());
			m_pTree->writeNode(nn.get());

			id_type cParent = pathBuffer.top(); pathBuffer.pop();
			NodePtr ptrN = m_pTree->readNode(cParent);
			Index* p = static_cast<Index*>(ptrN.get());
			p->adjustTree(n.get(), nn.get(), pathBuffer, overflowTable);
		}

		return true;
	}
}

void Node::reinsertData(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id, std::vector<uint32_t>& reinsert, std::vector<uint32_t>& keep)
{
	ReinsertEntry** v = new ReinsertEntry*[m_capacity + 1];

	m_pDataLength[m_children] = dataLength;
	m_pData[m_children] = pData;
	m_ptrMBR[m_children] = m_pTree->m_regionPool.acquire();
	*(m_ptrMBR[m_children]) = mbr;
	m_pIdentifier[m_children] = id;

	PointPtr nc = m_pTree->m_pointPool.acquire();
	m_nodeMBR.getCenter(*nc);
	PointPtr c = m_pTree->m_pointPool.acquire();

	for (uint32_t u32Child = 0; u32Child < m_capacity + 1; ++u32Child)
	{
		try
		{
			v[u32Child] = new ReinsertEntry(u32Child, 0.0);
		}
		catch (...)
		{
			for (uint32_t i = 0; i < u32Child; ++i) delete v[i];
			delete[] v;
			throw;
		}

		m_ptrMBR[u32Child]->getCenter(*c);

		// calculate relative distance of every entry from the node MBR (ignore square root.)
		for (uint32_t cDim = 0; cDim < m_nodeMBR.m_dimension; ++cDim)
		{
			double d = nc->m_pCoords[cDim] - c->m_pCoords[cDim];
			v[u32Child]->m_dist += d * d;
		}
	}

	// sort by increasing order of distances.
	::qsort(v, m_capacity + 1, sizeof(ReinsertEntry*), ReinsertEntry::compareReinsertEntry);

	uint32_t cReinsert = static_cast<uint32_t>(std::floor((m_capacity + 1) * m_pTree->m_reinsertFactor));

	uint32_t cCount;

	for (cCount = 0; cCount < m_capacity + 1; ++cCount)
	{
		if (cCount < m_capacity + 1 - cReinsert)
		{
			// Keep all but cReinsert nodes
			keep.push_back(v[cCount]->m_index);
		}
		else
		{
			// Remove cReinsert nodes which will be
			// reinserted into the tree. Since our array
			// is already sorted in ascending order this
			// matches the order suggested in the paper.
			reinsert.push_back(v[cCount]->m_index);
		}
		delete v[cCount];
	}

	delete[] v;
}

void Node::rtreeSplit(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id, std::vector<uint32_t>& group1, std::vector<uint32_t>& group2)
{
	uint32_t u32Child;
	uint32_t minimumLoad = static_cast<uint32_t>(std::floor(m_capacity * m_pTree->m_fillFactor));

	// use this mask array for marking visited entries.
	uint8_t* mask = new uint8_t[m_capacity + 1];
	memset(mask, 0, m_capacity + 1);

	// insert new data in the node for easier manipulation. Data arrays are always
	// by one larger than node capacity.
	m_pDataLength[m_capacity] = dataLength;
	m_pData[m_capacity] = pData;
	m_ptrMBR[m_capacity] = m_pTree->m_regionPool.acquire();
	*(m_ptrMBR[m_capacity]) = mbr;
	m_pIdentifier[m_capacity] = id;
	// m_totalDataLength does not need to be increased here.

	// initialize each group with the seed entries.
	uint32_t seed1, seed2;
	pickSeeds(seed1, seed2);

	group1.push_back(seed1);
	group2.push_back(seed2);

	mask[seed1] = 1;
	mask[seed2] = 1;

	// find MBR of each group.
	RegionPtr mbr1 = m_pTree->m_regionPool.acquire();
	*mbr1 = *(m_ptrMBR[seed1]);
	RegionPtr mbr2 = m_pTree->m_regionPool.acquire();
	*mbr2 = *(m_ptrMBR[seed2]);

	// count how many entries are left unchecked (exclude the seeds here.)
	uint32_t cRemaining = m_capacity + 1 - 2;

	while (cRemaining > 0)
	{
		if (minimumLoad - group1.size() == cRemaining)
		{
			// all remaining entries must be assigned to group1 to comply with minimun load requirement.
			for (u32Child = 0; u32Child < m_capacity + 1; ++u32Child)
			{
				if (mask[u32Child] == 0)
				{
					group1.push_back(u32Child);
					mask[u32Child] = 1;
					--cRemaining;
				}
			}
		}
		else if (minimumLoad - group2.size() == cRemaining)
		{
			// all remaining entries must be assigned to group2 to comply with minimun load requirement.
			for (u32Child = 0; u32Child < m_capacity + 1; ++u32Child)
			{
				if (mask[u32Child] == 0)
				{
					group2.push_back(u32Child);
					mask[u32Child] = 1;
					--cRemaining;
				}
			}
		}
		else
		{
			// For all remaining entries compute the difference of the cost of grouping an
			// entry in either group. When done, choose the entry that yielded the maximum
			// difference. In case of linear split, select any entry (e.g. the first one.)
			uint32_t sel;
			double md1 = 0.0, md2 = 0.0;
			double m = -std::numeric_limits<double>::max();
			double d1, d2, d;
			double a1 = mbr1->getArea();
			double a2 = mbr2->getArea();

			RegionPtr a = m_pTree->m_regionPool.acquire();
			RegionPtr b = m_pTree->m_regionPool.acquire();

			for (u32Child = 0; u32Child < m_capacity + 1; ++u32Child)
			{
				if (mask[u32Child] == 0)
				{
					mbr1->getCombinedRegion(*a, *(m_ptrMBR[u32Child]));
					d1 = a->getArea() - a1;
					mbr2->getCombinedRegion(*b, *(m_ptrMBR[u32Child]));
					d2 = b->getArea() - a2;
					d = std::abs(d1 - d2);

					if (d > m)
					{
						m = d;
						md1 = d1; md2 = d2;
						sel = u32Child;
					    if (m_pTree->m_treeVariant== RV_LINEAR || m_pTree->m_treeVariant == RV_RSTAR
					        || m_pTree->m_treeVariant == RV_BEST_SUBTREE || m_pTree->m_treeVariant == RV_MODEL_SUBTREE
					        || m_pTree->m_treeVariant == RV_BEST_SPLIT || m_pTree->m_treeVariant == RV_MODEL_SPLIT
					        || m_pTree->m_treeVariant == RV_BEST_BOTH || m_pTree->m_treeVariant == RV_MODEL_BOTH) break;
					}
				}
			}

			// determine the group where we should add the new entry.
			int32_t group = -1;

			if (md1 < md2)
			{
				group1.push_back(sel);
				group = 1;
			}
			else if (md2 < md1)
			{
				group2.push_back(sel);
				group = 2;
			}
			else if (a1 < a2)
			{
				group1.push_back(sel);
				group = 1;
			}
			else if (a2 < a1)
			{
				group2.push_back(sel);
				group = 2;
			}
			else if (group1.size() < group2.size())
			{
				group1.push_back(sel);
				group = 1;
			}
			else if (group2.size() < group1.size())
			{
				group2.push_back(sel);
				group = 2;
			}
			else
			{
				group1.push_back(sel);
				group = 1;
			}
			mask[sel] = 1;
			--cRemaining;
			if (group == 1)
			{
				mbr1->combineRegion(*(m_ptrMBR[sel]));
			}
			else
			{
				mbr2->combineRegion(*(m_ptrMBR[sel]));
			}
		}
	}

	delete[] mask;
}


void Node::enumerateRtreeSplit(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id, std::vector<SplitInfo> &split_infos, std::unordered_set< std::string > &existing_split_bbs_str_keys)
{
	uint32_t u32Child;
	// uint32_t minimumLoad = static_cast<uint32_t>(std::floor(m_capacity * m_pTree->m_fillFactor));
    uint32_t minimumLoad = static_cast<uint32_t>(std::floor((m_capacity + 1) * m_pTree->m_splitDistributionFactor));

    Region originalMBR = m_nodeMBR;
    double originalMBR_area = m_nodeMBR.getArea();

    std::vector<RTreeVariant> rvs = {RV_LINEAR, RV_QUADRATIC};

    for (RTreeVariant rv : rvs)
    {

        // use this mask array for marking visited entries.
        uint8_t* mask = new uint8_t[m_capacity + 1];
        memset(mask, 0, m_capacity + 1);

        // initialize each group with the seed entries.
        uint32_t seed1, seed2;
        pickSeeds_w_rv(seed1, seed2, rv);

        std::vector<uint32_t> group1;
        std::vector<uint32_t> group2;

        group1.push_back(seed1);
        group2.push_back(seed2);

        mask[seed1] = 1;
        mask[seed2] = 1;

        // find MBR of each group.
        RegionPtr mbr1 = m_pTree->m_regionPool.acquire();
        *mbr1 = *(m_ptrMBR[seed1]);
        RegionPtr mbr2 = m_pTree->m_regionPool.acquire();
        *mbr2 = *(m_ptrMBR[seed2]);

        // count how many entries are left unchecked (exclude the seeds here.)
        uint32_t cRemaining = m_capacity + 1 - 2;

        while (cRemaining > 0)
        {
            if (minimumLoad - group1.size() == cRemaining)
            {
                // all remaining entries must be assigned to group1 to comply with minimun load requirement.
                for (u32Child = 0; u32Child < m_capacity + 1; ++u32Child)
                {
                    if (mask[u32Child] == 0)
                    {
                        group1.push_back(u32Child);
                        mask[u32Child] = 1;
                        --cRemaining;

                        mbr1->combineRegion(*(m_ptrMBR[u32Child]));
                    }
                }
            }
            else if (minimumLoad - group2.size() == cRemaining)
            {
                // all remaining entries must be assigned to group2 to comply with minimun load requirement.
                for (u32Child = 0; u32Child < m_capacity + 1; ++u32Child)
                {
                    if (mask[u32Child] == 0)
                    {
                        group2.push_back(u32Child);
                        mask[u32Child] = 1;
                        --cRemaining;

                        mbr2->combineRegion(*(m_ptrMBR[u32Child]));
                    }
                }
            }
            else
            {
                // For all remaining entries compute the difference of the cost of grouping an
                // entry in either group. When done, choose the entry that yielded the maximum
                // difference. In case of linear split, select any entry (e.g. the first one.)
                uint32_t sel;
                double md1 = 0.0, md2 = 0.0;
                double m = -std::numeric_limits<double>::max();
                double d1, d2, d;
                double a1 = mbr1->getArea();
                double a2 = mbr2->getArea();

                RegionPtr a = m_pTree->m_regionPool.acquire();
                RegionPtr b = m_pTree->m_regionPool.acquire();

                for (u32Child = 0; u32Child < m_capacity + 1; ++u32Child)
                {
                    if (mask[u32Child] == 0)
                    {
                        mbr1->getCombinedRegion(*a, *(m_ptrMBR[u32Child]));
                        d1 = a->getArea() - a1;
                        mbr2->getCombinedRegion(*b, *(m_ptrMBR[u32Child]));
                        d2 = b->getArea() - a2;
                        d = std::abs(d1 - d2);

                        if (d > m)
                        {
                            m = d;
                            md1 = d1; md2 = d2;
                            sel = u32Child;
                            if (rv == RV_LINEAR || rv == RV_RSTAR) break;
                        }
                    }
                }

                // determine the group where we should add the new entry.
                int32_t group = -1;

                if (md1 < md2)
                {
                    group1.push_back(sel);
                    group = 1;
                }
                else if (md2 < md1)
                {
                    group2.push_back(sel);
                    group = 2;
                }
                else if (a1 < a2)
                {
                    group1.push_back(sel);
                    group = 1;
                }
                else if (a2 < a1)
                {
                    group2.push_back(sel);
                    group = 2;
                }
                else if (group1.size() < group2.size())
                {
                    group1.push_back(sel);
                    group = 1;
                }
                else if (group2.size() < group1.size())
                {
                    group2.push_back(sel);
                    group = 2;
                }
                else
                {
                    group1.push_back(sel);
                    group = 1;
                }
                mask[sel] = 1;
                --cRemaining;
                if (group == 1)
                {
                    mbr1->combineRegion(*(m_ptrMBR[sel]));
                }
                else
                {
                    mbr2->combineRegion(*(m_ptrMBR[sel]));
                }
            }
        }

        std::string s1 = std::move(split_bbs_to_string(*mbr1, *mbr2));
        std::string s2 = std::move(split_bbs_to_string(*mbr2, *mbr1));
        if ((!existing_split_bbs_str_keys.empty()) && (existing_split_bbs_str_keys.find(s1) != existing_split_bbs_str_keys.end() || existing_split_bbs_str_keys.find(s2) != existing_split_bbs_str_keys.end()))
        {
            continue;
        }
        else
        {
            existing_split_bbs_str_keys.insert(s1);
            existing_split_bbs_str_keys.insert(s2);
        }

        split_infos.emplace_back();
        SplitInfo &split_info = split_infos.back();

        split_info.from_rstar = false;
        split_info.splitAxis = m_pTree->m_dimension;
        split_info.sortOrder = 2;

        split_info.bb1 = *mbr1;
        split_info.bb2 = *mbr2;

        double o = split_info.bb1.getIntersectingArea(split_info.bb2);
        split_info.overlap = o;

        double bb1_area = split_info.bb1.getArea();
        double bb2_area = split_info.bb2.getArea();
        split_info.bb1_area = bb1_area;
        split_info.bb2_area = bb2_area;

        double bb1_margin = split_info.bb1.getMargin();
        double bb2_margin = split_info.bb2.getMargin();
        split_info.bb1_margin = bb1_margin;
        split_info.bb2_margin = bb2_margin;

        double margin = bb1_margin + bb2_margin;
        double area = bb1_area + bb2_area;
        split_info.area = area;
        split_info.margin = margin;

        split_info.increased_area = bb1_area + bb2_area - originalMBR_area;
        split_info.get_candi_rep_points();

        split_info.group1 = std::move(group1);
        split_info.group2 = std::move(group2);

        delete[] mask;
    }
}

std::string SpatialIndex::RTree::split_bbs_to_string(Region &bb1, Region &bb2)
{
    // std::string s = std::format("{.4f}", bb1.m_pLow[0]);
    char buffer[40];

    std::string res = "";
    for (uint32_t d = 0; d < bb1.m_dimension; ++d)
    {
        snprintf(buffer, 40, "%.5f,%.5f,", bb1.m_pLow[d], bb1.m_pHigh[d]);
        std::string s(buffer);
        res += s;
    }
    for (uint32_t d = 0; d < bb2.m_dimension; ++d)
    {
        snprintf(buffer, 40, "%.5f,%.5f,", bb2.m_pLow[d], bb2.m_pHigh[d]);
        std::string s(buffer);
        res += s;
    }
    return res;
}


void Node::enumerateRstarSplit(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id, std::vector< SplitInfo >& split_infos, std::unordered_set< std::string > &existing_split_bbs_str_keys)
{
	RstarSplitEntry** dataLow = nullptr;

	try
	{
		dataLow = new RstarSplitEntry*[m_capacity + 1];
	}
	catch (...)
	{
		delete[] dataLow;
		throw;
	}


	uint32_t nodeSPF = static_cast<uint32_t>(
		std::floor((m_capacity + 1) * m_pTree->m_splitDistributionFactor));
	uint32_t splitDistribution = (m_capacity + 1) - (2 * nodeSPF) + 2;

	uint32_t u32Child = 0, cDim, cIndex;

	for (u32Child = 0; u32Child <= m_capacity; ++u32Child)
	{
		try
		{
			dataLow[u32Child] = new RstarSplitEntry(m_ptrMBR[u32Child].get(), u32Child, 0);
		}
		catch (...)
		{
			for (uint32_t i = 0; i < u32Child; ++i) delete dataLow[i];
			delete[] dataLow;
			throw;
		}
	}

	uint32_t splitAxis = std::numeric_limits<uint32_t>::max();
	uint32_t sortOrder = std::numeric_limits<uint32_t>::max();

    Region originalMBR = m_nodeMBR;
    double originalMBR_area = m_nodeMBR.getArea();

    std::vector< std::vector<double> > best_cost_all_axes;
    best_cost_all_axes.emplace_back();
    best_cost_all_axes.emplace_back();
    std::vector<double> best_cost_along_axes_low(m_pTree->m_dimension, std::numeric_limits<double>::max());
    std::vector<double> best_cost_along_axes_high(m_pTree->m_dimension, std::numeric_limits<double>::max());
    best_cost_all_axes[0] = std::move(best_cost_along_axes_low);
    best_cost_all_axes[1] = std::move(best_cost_along_axes_high);

    std::unordered_set< std::string > existing_split_str_keys;

    for (splitAxis = 0; splitAxis < m_pTree->m_dimension; ++splitAxis)
    {
        for (u32Child = 0; u32Child <= m_capacity; ++u32Child)
        {
            dataLow[u32Child]->m_sortDim = splitAxis;
        }

        for (sortOrder = 0; sortOrder <= 1; ++sortOrder)
        {
            ::qsort(dataLow, m_capacity + 1, sizeof(RstarSplitEntry*), (sortOrder == 0) ? RstarSplitEntry::compareLow : RstarSplitEntry::compareHigh);


            uint32_t splitPoint = std::numeric_limits<uint32_t>::max();

            for (u32Child = 1; u32Child <= splitDistribution; ++u32Child)
            {


                uint32_t l = nodeSPF - 1 + u32Child;
                Region bb1 = *(dataLow[0]->m_pRegion);

                for (cIndex = 1; cIndex < l; ++cIndex)
                {
                    bb1.combineRegion(*(dataLow[cIndex]->m_pRegion));
                }

                Region bb2 = *(dataLow[l]->m_pRegion);

                for (cIndex = l + 1; cIndex <= m_capacity; ++cIndex)
                {
                    bb2.combineRegion(*(dataLow[cIndex]->m_pRegion));
                }

                std::string s1 = std::move(split_bbs_to_string(bb1, bb2));
                std::string s2 = std::move(split_bbs_to_string(bb2, bb1));
                if ((!existing_split_bbs_str_keys.empty()) && (existing_split_bbs_str_keys.find(s1) != existing_split_bbs_str_keys.end() || existing_split_bbs_str_keys.find(s2) != existing_split_bbs_str_keys.end()))
                {
                    continue;
                }
                else
                {
                    existing_split_bbs_str_keys.insert(s1);
                    existing_split_bbs_str_keys.insert(s2);
                }

                split_infos.emplace_back();
                SplitInfo &split_info = split_infos.back();

                splitPoint = u32Child;

                uint32_t l1 = nodeSPF - 1 + splitPoint;

                if (m_capacity < 64)
                {
                    uint64_t key1 = 0, key2 = 0;
                    const uint64_t one = 1;
                    for (cIndex = 0; cIndex < l1; ++cIndex)
                    {
                        uint32_t index = dataLow[cIndex]->m_index;
                        split_info.group1.push_back(index);
                        key1 |= (one << index);
                        // delete dataLow[cIndex];
                    }

                    for (cIndex = l1; cIndex <= m_capacity; ++cIndex)
                    {
                        uint32_t index = dataLow[cIndex]->m_index;
                        split_info.group2.push_back(index);
                        key2 |= (one << index);
                        // delete dataLow[cIndex];
                    }
                }
                else
                {
                    std::string key1(m_capacity + 1, '0');
                    std::string key2(key1);
                    for (cIndex = 0; cIndex < l1; ++cIndex)
                    {
                        uint32_t index = dataLow[cIndex]->m_index;
                        split_info.group1.push_back(index);
                        // delete dataLow[cIndex];
                        key1[index] = '1';
                    }

                    for (cIndex = l1; cIndex <= m_capacity; ++cIndex)
                    {
                        uint32_t index = dataLow[cIndex]->m_index;
                        split_info.group2.push_back(index);
                        key2[index] = '1';
                        // delete dataLow[cIndex];
                    }
                }
                double o = bb1.getIntersectingArea(bb2);
                split_info.overlap = o;

                double bb1_area = bb1.getArea();
                double bb2_area = bb2.getArea();
                split_info.bb1_area = bb1_area;
                split_info.bb2_area = bb2_area;

                double bb1_margin = bb1.getMargin();
                double bb2_margin = bb2.getMargin();
                split_info.bb1_margin = bb1_margin;
                split_info.bb2_margin = bb2_margin;

                double margin = bb1_margin + bb2_margin;
                double area = bb1_area + bb2_area;

                split_info.area = area;
                split_info.margin = margin;

                split_info.from_rstar = true;
                split_info.splitAxis = splitAxis;
                split_info.sortOrder = sortOrder;

                split_info.bb1 = bb1;
                split_info.bb2 = bb2;
                split_info.increased_area = bb1_area + bb2_area - originalMBR_area;
                split_info.get_candi_rep_points();
            } // for (u32Child)

        }
    }


    for (cIndex = 0; cIndex <= m_capacity; ++cIndex)
    {
        delete dataLow[cIndex];
    }

	delete[] dataLow;


}

void SpatialIndex::RTree::get_padding_split_info(Region &m_nodeMBR, SplitInfo &split_info)
{
    // std::cout << "1. It is OK here." << std::endl;
    split_info.bb1 = m_nodeMBR;
    split_info.bb2 = m_nodeMBR;
    // std::cout << "2. It is OK here." << std::endl;

    double bb1_area = split_info.bb1.getArea();
    // double bb2_area = split_info.bb2.getArea();

    // double o = split_info.bb1.getIntersectingArea(split_info.bb2);
    split_info.overlap = bb1_area;
    split_info.bb1_area = bb1_area;
    split_info.bb2_area = bb1_area;
    // std::cout << "3. It is OK here." << std::endl;

    double bb1_margin = split_info.bb1.getMargin();
    // double bb2_margin = split_info.bb2.getMargin();
    split_info.bb1_margin = bb1_margin;
    split_info.bb2_margin = bb1_margin;

    double margin = bb1_margin * 2;
    double area = bb1_area * 2;
    split_info.area = area;
    split_info.margin = margin;

    split_info.from_rstar = false;
    split_info.splitAxis = m_nodeMBR.m_dimension + 2;
    split_info.sortOrder = m_nodeMBR.m_dimension + 2;

    split_info.increased_area = bb1_area;
    // std::cout << "4. It is OK here." << std::endl;
    split_info.get_candi_rep_points();
    // std::cout << "5. It is OK here. bb1.m_dimension = " << split_info.bb1.m_dimension << std::endl;
}


void Node::enumerateCandidateSplit(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id,
    std::vector< SplitInfo >& split_infos)
{
    m_pDataLength[m_capacity] = dataLength;
    m_pData[m_capacity] = pData;
    m_ptrMBR[m_capacity] = m_pTree->m_regionPool.acquire();
    *(m_ptrMBR[m_capacity]) = mbr;
    m_pIdentifier[m_capacity] = id;
    // m_totalDataLength does not need to be increased here.

    std::unordered_set< std::string > existing_split_bbs_str_keys;
    enumerateRstarSplit(dataLength, pData, mbr, id, split_infos, existing_split_bbs_str_keys);
    // enumerateRtreeSplit(dataLength, pData, mbr, id, split_infos, existing_split_bbs_str_keys);
    // if (m_level > 0)
    // {
    //     std::sort(split_infos.begin(), split_infos.end(), SplitInfo::compare);
    // }
}

void Node::rstarSplit(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id, std::vector<uint32_t>& group1, std::vector<uint32_t>& group2)
{
	RstarSplitEntry** dataLow = nullptr;
	RstarSplitEntry** dataHigh = nullptr;

	try
	{
		dataLow = new RstarSplitEntry*[m_capacity + 1];
		dataHigh = new RstarSplitEntry*[m_capacity + 1];
	}
	catch (...)
	{
		delete[] dataLow;
		throw;
	}

	m_pDataLength[m_capacity] = dataLength;
	m_pData[m_capacity] = pData;
	m_ptrMBR[m_capacity] = m_pTree->m_regionPool.acquire();
	*(m_ptrMBR[m_capacity]) = mbr;
	m_pIdentifier[m_capacity] = id;
	// m_totalDataLength does not need to be increased here.

	uint32_t nodeSPF = static_cast<uint32_t>(
		std::floor((m_capacity + 1) * m_pTree->m_splitDistributionFactor));
	uint32_t splitDistribution = (m_capacity + 1) - (2 * nodeSPF) + 2;

	uint32_t u32Child = 0, cDim, cIndex;

	for (u32Child = 0; u32Child <= m_capacity; ++u32Child)
	{
		try
		{
			dataLow[u32Child] = new RstarSplitEntry(m_ptrMBR[u32Child].get(), u32Child, 0);
		}
		catch (...)
		{
			for (uint32_t i = 0; i < u32Child; ++i) delete dataLow[i];
			delete[] dataLow;
			delete[] dataHigh;
			throw;
		}

		dataHigh[u32Child] = dataLow[u32Child];
	}

	double minimumMargin = std::numeric_limits<double>::max();
	uint32_t splitAxis = std::numeric_limits<uint32_t>::max();
	uint32_t sortOrder = std::numeric_limits<uint32_t>::max();

	// chooseSplitAxis.
	for (cDim = 0; cDim < m_pTree->m_dimension; ++cDim)
	{
		::qsort(dataLow, m_capacity + 1, sizeof(RstarSplitEntry*), RstarSplitEntry::compareLow);
		::qsort(dataHigh, m_capacity + 1, sizeof(RstarSplitEntry*), RstarSplitEntry::compareHigh);

		// calculate sum of margins and overlap for all distributions.
		double marginl = 0.0;
		double marginh = 0.0;

		Region bbl1, bbl2, bbh1, bbh2;

		for (u32Child = 1; u32Child <= splitDistribution; ++u32Child)
		{
			uint32_t l = nodeSPF - 1 + u32Child;

			bbl1 = *(dataLow[0]->m_pRegion);
			bbh1 = *(dataHigh[0]->m_pRegion);

			for (cIndex = 1; cIndex < l; ++cIndex)
			{
				bbl1.combineRegion(*(dataLow[cIndex]->m_pRegion));
				bbh1.combineRegion(*(dataHigh[cIndex]->m_pRegion));
			}

			bbl2 = *(dataLow[l]->m_pRegion);
			bbh2 = *(dataHigh[l]->m_pRegion);

			for (cIndex = l + 1; cIndex <= m_capacity; ++cIndex)
			{
				bbl2.combineRegion(*(dataLow[cIndex]->m_pRegion));
				bbh2.combineRegion(*(dataHigh[cIndex]->m_pRegion));
			}

			marginl += bbl1.getMargin() + bbl2.getMargin();
			marginh += bbh1.getMargin() + bbh2.getMargin();
		} // for (u32Child)

		double margin = std::min(marginl, marginh);

		// keep minimum margin as split axis.
		if (margin < minimumMargin)
		{
			minimumMargin = margin;
			splitAxis = cDim;
			sortOrder = (marginl < marginh) ? 0 : 1;
		}

		// increase the dimension according to which the data entries should be sorted.
		for (u32Child = 0; u32Child <= m_capacity; ++u32Child)
		{
			dataLow[u32Child]->m_sortDim = cDim + 1;
		}
	} // for (cDim)

	for (u32Child = 0; u32Child <= m_capacity; ++u32Child)
	{
		dataLow[u32Child]->m_sortDim = splitAxis;
	}

	::qsort(dataLow, m_capacity + 1, sizeof(RstarSplitEntry*), (sortOrder == 0) ? RstarSplitEntry::compareLow : RstarSplitEntry::compareHigh);

	double ma = std::numeric_limits<double>::max();
	double mo = std::numeric_limits<double>::max();
	uint32_t splitPoint = std::numeric_limits<uint32_t>::max();

	Region bb1, bb2;

	for (u32Child = 1; u32Child <= splitDistribution; ++u32Child)
	{
		uint32_t l = nodeSPF - 1 + u32Child;

		bb1 = *(dataLow[0]->m_pRegion);

		for (cIndex = 1; cIndex < l; ++cIndex)
		{
			bb1.combineRegion(*(dataLow[cIndex]->m_pRegion));
		}

		bb2 = *(dataLow[l]->m_pRegion);

		for (cIndex = l + 1; cIndex <= m_capacity; ++cIndex)
		{
			bb2.combineRegion(*(dataLow[cIndex]->m_pRegion));
		}

		double o = bb1.getIntersectingArea(bb2);

		if (o < mo)
		{
			splitPoint = u32Child;
			mo = o;
			ma = bb1.getArea() + bb2.getArea();
		}
		else if (o == mo)
		{
			double a = bb1.getArea() + bb2.getArea();

			if (a < ma)
			{
				splitPoint = u32Child;
				ma = a;
			}
		}
	} // for (u32Child)

	uint32_t l1 = nodeSPF - 1 + splitPoint;

	for (cIndex = 0; cIndex < l1; ++cIndex)
	{
		group1.push_back(dataLow[cIndex]->m_index);
		delete dataLow[cIndex];
	}

	for (cIndex = l1; cIndex <= m_capacity; ++cIndex)
	{
		group2.push_back(dataLow[cIndex]->m_index);
		delete dataLow[cIndex];
	}

	delete[] dataLow;
	delete[] dataHigh;
}

void Node::simulate_rstarSplit(Region& mbr, simulateSplitInfo &ssi) //Region &best_bb1, Region &best_bb2, double &best_margin, double &best_overlap, double &best_area) //, std::vector<uint32_t>& group1, std::vector<uint32_t>& group2
{
	RstarSplitEntry** dataLow = nullptr;
	RstarSplitEntry** dataHigh = nullptr;

	try
	{
		dataLow = new RstarSplitEntry*[m_capacity + 1];
		dataHigh = new RstarSplitEntry*[m_capacity + 1];
	}
	catch (...)
	{
		delete[] dataLow;
		throw;
	}

	// m_pDataLength[m_capacity] = dataLength;
	// m_pData[m_capacity] = pData;
	m_ptrMBR[m_capacity] = m_pTree->m_regionPool.acquire();
	*(m_ptrMBR[m_capacity]) = mbr;
	// m_pIdentifier[m_capacity] = id;
	// m_totalDataLength does not need to be increased here.

	uint32_t nodeSPF = static_cast<uint32_t>(
		std::floor((m_capacity + 1) * m_pTree->m_splitDistributionFactor));
	uint32_t splitDistribution = (m_capacity + 1) - (2 * nodeSPF) + 2;

	uint32_t u32Child = 0, cDim, cIndex;

	for (u32Child = 0; u32Child <= m_capacity; ++u32Child)
	{
		try
		{
			dataLow[u32Child] = new RstarSplitEntry(m_ptrMBR[u32Child].get(), u32Child, 0);
		}
		catch (...)
		{
			for (uint32_t i = 0; i < u32Child; ++i) delete dataLow[i];
			delete[] dataLow;
			delete[] dataHigh;
			throw;
		}

		dataHigh[u32Child] = dataLow[u32Child];
	}

	double minimumMargin = std::numeric_limits<double>::max();
	uint32_t splitAxis = std::numeric_limits<uint32_t>::max();
	uint32_t sortOrder = std::numeric_limits<uint32_t>::max();

	// chooseSplitAxis.
	for (cDim = 0; cDim < m_pTree->m_dimension; ++cDim)
	{
		::qsort(dataLow, m_capacity + 1, sizeof(RstarSplitEntry*), RstarSplitEntry::compareLow);
		::qsort(dataHigh, m_capacity + 1, sizeof(RstarSplitEntry*), RstarSplitEntry::compareHigh);

		// calculate sum of margins and overlap for all distributions.
		double marginl = 0.0;
		double marginh = 0.0;

		Region bbl1, bbl2, bbh1, bbh2;

		for (u32Child = 1; u32Child <= splitDistribution; ++u32Child)
		{
			uint32_t l = nodeSPF - 1 + u32Child;

			bbl1 = *(dataLow[0]->m_pRegion);
			bbh1 = *(dataHigh[0]->m_pRegion);

			for (cIndex = 1; cIndex < l; ++cIndex)
			{
				bbl1.combineRegion(*(dataLow[cIndex]->m_pRegion));
				bbh1.combineRegion(*(dataHigh[cIndex]->m_pRegion));
			}

			bbl2 = *(dataLow[l]->m_pRegion);
			bbh2 = *(dataHigh[l]->m_pRegion);

			for (cIndex = l + 1; cIndex <= m_capacity; ++cIndex)
			{
				bbl2.combineRegion(*(dataLow[cIndex]->m_pRegion));
				bbh2.combineRegion(*(dataHigh[cIndex]->m_pRegion));
			}

			marginl += bbl1.getMargin() + bbl2.getMargin();
			marginh += bbh1.getMargin() + bbh2.getMargin();
		} // for (u32Child)

		double margin = std::min(marginl, marginh);

		// keep minimum margin as split axis.
		if (margin < minimumMargin)
		{
			minimumMargin = margin;
			splitAxis = cDim;
			sortOrder = (marginl < marginh) ? 0 : 1;
		}

		// increase the dimension according to which the data entries should be sorted.
		for (u32Child = 0; u32Child <= m_capacity; ++u32Child)
		{
			dataLow[u32Child]->m_sortDim = cDim + 1;
		}
	} // for (cDim)

	for (u32Child = 0; u32Child <= m_capacity; ++u32Child)
	{
		dataLow[u32Child]->m_sortDim = splitAxis;
	}

	::qsort(dataLow, m_capacity + 1, sizeof(RstarSplitEntry*), (sortOrder == 0) ? RstarSplitEntry::compareLow : RstarSplitEntry::compareHigh);

    ssi.splitAxis = splitAxis;
    ssi.sortOrder = sortOrder;

	double ma = std::numeric_limits<double>::max();
	double mo = std::numeric_limits<double>::max();
	uint32_t splitPoint = std::numeric_limits<uint32_t>::max();

    Region originalMBR = m_nodeMBR;
    Region bb1, bb2;
	for (u32Child = 1; u32Child <= splitDistribution; ++u32Child)
	{
		uint32_t l = nodeSPF - 1 + u32Child;

		bb1 = *(dataLow[0]->m_pRegion);

		for (cIndex = 1; cIndex < l; ++cIndex)
		{
			bb1.combineRegion(*(dataLow[cIndex]->m_pRegion));
		}

		bb2 = *(dataLow[l]->m_pRegion);

		for (cIndex = l + 1; cIndex <= m_capacity; ++cIndex)
		{
			bb2.combineRegion(*(dataLow[cIndex]->m_pRegion));
		}

		double o = bb1.getIntersectingArea(bb2);

		if (o < mo)
		{
			splitPoint = u32Child;
			mo = o;
			ma = bb1.getArea() + bb2.getArea();
		    ssi.bb1 = bb1;
		    ssi.bb2 = bb2;
		}
		else if (o == mo)
		{
			double a = bb1.getArea() + bb2.getArea();

			if (a < ma)
			{
				splitPoint = u32Child;
			    ssi.bb1 = bb1;
			    ssi.bb2 = bb2;
				ma = a;
			}
		}
	} // for (u32Child)

    ssi.margin = minimumMargin;
    ssi.overlap = mo;
    ssi.area = ma;
    ssi.increased_area = ma - originalMBR.getArea();

	delete[] dataLow;
	delete[] dataHigh;
}

double Node::getIntersectingAreaBetweenTwoAndOthers(const Region& bb1, const Region &bb2, id_type child_identifier)
{
    double o = bb1.getIntersectingArea(bb2);
    // Region intersect = bb1.getIntersectingRegion(bb2);
    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        if (m_pIdentifier[cChild] != child_identifier)
        {
            o += bb1.getIntersectingArea(*m_ptrMBR[cChild]);
            o += bb2.getIntersectingArea(*m_ptrMBR[cChild]);
            // o -= intersect.getIntersectingArea(*m_ptrMBR[cChild]);
        }
    }
    return o;
}

double Node::getIntersectingAreaBetweenOneAndOthers(const Region& bb, id_type child_identifier)
{
    double o = 0;
    for (uint32_t cChild = 0; cChild < m_children; ++cChild)
    {
        if (m_pIdentifier[cChild] != child_identifier)
        {
            o += bb.getIntersectingArea(*m_ptrMBR[cChild]);
        }
    }
    return o;
}


uint64_t Node::calc_split_cost_w_queries(std::vector<id_type>& insert_path, std::vector< SplitInfo >& split_infos, uint64_t k, std::vector<Region> &queries, std::vector<uint64_t> &io_cost_each_query)
{
    io_cost_each_query = std::move(std::vector< uint64_t >(queries.size(), 0));
    SplitInfo &split_info = split_infos[k];
    Region original_bb = m_nodeMBR;
    Region bb1 = split_info.bb1;
    Region bb2 = split_info.bb2;
    id_type child_identifier = this->m_identifier;

    uint64_t cost = 0;
    for (uint32_t i = 0; i < insert_path.size(); ++i)
    {

        id_type cParent = insert_path[i];
        NodePtr parent_node = m_pTree->readNode(cParent);
        assert(parent_node->m_level == m_level + 1 + i);
        if (parent_node->m_level != m_level + 1 + i)
        {
            int left = parent_node->m_level;
            int right = m_level + 1 + i;
            throw Tools::IsNotEqualException<int>("Node::calc_split_cost", "parent_node->m_level", "m_level + 1 + i", left, right);
        }

        for (size_t j = 0; j < queries.size(); ++j)
        {
            Region &query = queries[j];

            uint64_t cost_j = static_cast<uint64_t>(bb1.intersectsRegion(query)) + static_cast<uint64_t>(bb2.intersectsRegion(query));
            io_cost_each_query[j] += cost_j;
            cost += cost_j;
        }
        break;

        if (parent_node->m_children < parent_node->m_capacity)
        {
            bool flag = false;

            for (uint32_t cChild = 0; cChild < parent_node->m_children; ++cChild)
            {
                if (parent_node->m_pIdentifier[cChild] == child_identifier)
                {
                    flag = true;
                }
            }
            if (!flag)
            {
                throw Tools::BasicException("In Node::calc_split_cost, flag should be true. Now it is false.");
            }
            break;

        }
        else
        {
            Index parent_backup(m_pTree, -1, parent_node->m_level);
            parent_backup.simpleClone(*parent_node);
            bool flag = false;
            for (uint32_t cChild = 0; cChild < parent_backup.m_children; ++cChild)
            {
                if (parent_backup.m_pIdentifier[cChild] == child_identifier)
                {
                    *(parent_backup.m_ptrMBR[cChild]) = bb1;
                    flag = true;
                }
            }
            if (!flag)
            {
                throw Tools::BasicException("In Node::calc_split_cost, flag should be true. Now it is false.");
            }

            // simulate the rstar split process at the node parent_backup
            // return new bb1, bb2, child_identifier
            Region last_bb2 = bb2;
            simulateSplitInfo ssi;
            parent_backup.simulate_rstarSplit(last_bb2, ssi);
            bb1 = ssi.bb1;
            bb2 = ssi.bb2;

            child_identifier = parent_node->m_identifier;
            original_bb = parent_node->m_nodeMBR;
        }
    }

    return cost;
}


void SpatialIndex::RTree::get_split_geometric_properties(std::vector< SplitInfo > &split_infos, std::vector< double > &geometric_stats)
{
    uint64_t num_splits = split_infos.size() - 1;
    double min_overlap = std::numeric_limits<double>::max();
    double min_area = std::numeric_limits<double>::max();
    double min_margin = std::numeric_limits<double>::max();
    double min_total_area = std::numeric_limits<double>::max();
    double min_total_margin = std::numeric_limits<double>::max();

    double max_overlap = std::numeric_limits<double>::min();
    double max_area = std::numeric_limits<double>::min();
    double max_margin = std::numeric_limits<double>::min();
    double max_total_area = std::numeric_limits<double>::max();
    double max_total_margin = std::numeric_limits<double>::max();

    double avg_area = 0;
    double avg_margin = 0;
    double avg_total_area = 0;
    double avg_total_margin = 0;


    for (uint64_t s = 0; s < num_splits; ++s)
    {
        SplitInfo &split_info = split_infos[s];

        if (min_overlap > split_info.overlap)
        {
            min_overlap = split_info.overlap;
        }
        if (max_overlap < split_info.overlap)
        {
            max_overlap = split_info.overlap;
        }


        if (min_area > split_info.bb1_area)
        {
            min_area = split_info.bb1_area;
        }
        if (min_area > split_info.bb2_area)
        {
            min_area = split_info.bb2_area;
        }

        if (max_area < split_info.bb1_area)
        {
            max_area = split_info.bb1_area;
        }
        if (max_area < split_info.bb2_area)
        {
            max_area = split_info.bb2_area;
        }


        if (min_margin > split_info.bb1_margin)
        {
            min_margin = split_info.bb2_margin;
        }
        if (min_margin > split_info.bb2_margin)
        {
            min_margin = split_info.bb2_margin;
        }

        if (max_margin < split_info.bb1_margin)
        {
            max_margin = split_info.bb1_margin;
        }
        if (max_margin < split_info.bb2_margin)
        {
            max_margin = split_info.bb2_margin;
        }


        if (min_total_area > split_info.area)
        {
            min_total_area = split_info.area;
        }
        if (min_total_margin > split_info.margin)
        {
            min_total_margin = split_info.margin;
        }

        if (max_total_area < split_info.area)
        {
            max_total_area = split_info.area;
        }
        if (max_total_margin < split_info.margin)
        {
            max_total_margin = split_info.margin;
        }
        avg_area += split_info.area;
        avg_margin += split_info.margin;
        avg_total_area += split_info.area;
        avg_total_margin += split_info.margin;
    }
    avg_area /= (static_cast<double>(num_splits * 2));
    avg_margin /= (static_cast<double>(num_splits * 2));
    avg_total_area /= (static_cast<double>(num_splits));
    avg_total_margin /= (static_cast<double>(num_splits));

    std::vector<double> _stats = {min_overlap, min_area, min_margin, min_total_area, min_total_margin,
        max_overlap, max_area, max_margin, max_total_area, max_total_margin,
        avg_area, avg_margin, avg_total_area, avg_total_margin};
    geometric_stats = std::move(_stats);

    SplitInfo &padding_split_info = split_infos[num_splits];

    double max_margin_feature = 0;
    double max_area_feature = 0;
    // get geometric features
    for (uint64_t s = 0; s < num_splits; ++s)
    {
        SplitInfo &split_info = split_infos[s];
        std::vector<double> &geometric_properties = split_info.geometric_properties;
        double overlap_feature = 0;
        if (split_info.overlap > 0)
        {
            overlap_feature = split_info.overlap / max_overlap;
        }
        geometric_properties.emplace_back(overlap_feature);
        double bb1_margin_feature = split_info.bb1_margin / avg_total_margin;
        double bb2_margin_feature = split_info.bb2_margin / avg_total_margin;
        double bb1_area_feature = split_info.bb1_area / avg_total_area;
        double bb2_area_feature = split_info.bb2_area / avg_total_area;
        if (bb1_margin_feature > max_margin_feature)
        {
            max_margin_feature = bb1_margin_feature;
        }
        if (bb2_margin_feature > max_margin_feature)
        {
            max_margin_feature = bb2_margin_feature;
        }
        if (bb1_area_feature > max_area_feature)
        {
            max_area_feature = bb1_area_feature;
        }
        if (bb2_area_feature > max_area_feature)
        {
            max_area_feature = bb2_area_feature;
        }
        geometric_properties.emplace_back(bb1_margin_feature);
        geometric_properties.emplace_back(bb2_margin_feature);
        geometric_properties.emplace_back(bb1_area_feature);
        geometric_properties.emplace_back(bb2_area_feature);
    }
    std::vector<double> &geometric_properties = padding_split_info.geometric_properties;
    geometric_properties.emplace_back(2.0);
    double margin_feature = padding_split_info.bb1_margin / avg_total_margin;
    double area_feature = padding_split_info.bb1_area / avg_total_area;
    if (margin_feature > 2 * max_margin_feature)
    {
        margin_feature = 2 * margin_feature;
    }
    if (area_feature > 2 * max_area_feature)
    {
        area_feature = 2 * area_feature;
    }
    geometric_properties.emplace_back(margin_feature);
    geometric_properties.emplace_back(margin_feature);
    geometric_properties.emplace_back(area_feature);
    geometric_properties.emplace_back(area_feature);
}


int64_t SpatialIndex::RTree::get_split_scores(std::vector< SplitInfo > &split_infos, std::vector< double > &geometric_stats,
    std::vector< std::vector<uint64_t> > &io_cost_each_split, std::vector< double > &split_scores, bool split_likelihood_as_score, bool print_info)
{
    uint64_t num_splits = io_cost_each_split.size() - 1;
    uint64_t num_queries = io_cost_each_split[0].size();

    std::vector<uint64_t> best_io_each_query(num_queries, std::numeric_limits<uint64_t>::max());
    std::vector<uint64_t> worst_io_each_query(num_queries, 0);

    std::vector<uint64_t> num_best_performance_each_split(num_splits + 1, 0);

    // statistics of best io for each query
    for (uint64_t q = 0; q < num_queries; ++q)
    {
        uint64_t best_io = std::numeric_limits<uint64_t>::max();
        uint64_t worst_io = 0;
        for (uint64_t s = 0; s < num_splits; ++s)
        {
            if (io_cost_each_split[s][q] < best_io)
            {
                best_io = io_cost_each_split[s][q];
            }
            if (io_cost_each_split[s][q] > worst_io)
            {
                worst_io = io_cost_each_split[s][q];
            }
        }
        best_io_each_query[q] = best_io;
        worst_io_each_query[q] = worst_io;
    }

    std::vector<uint64_t> total_cost_each_split(num_splits + 1, 0);
    uint64_t total_best_cost = 0;
    uint64_t total_for_likelihood = 0;
    uint64_t count = 0;
    for (uint64_t q = 0; q < num_queries; ++q)
    {
        uint64_t best_io = best_io_each_query[q];
        uint64_t worst_io = worst_io_each_query[q];
        if (worst_io <= best_io)
        {
            continue;
        }
        ++count;
        total_best_cost += best_io;
        for (uint64_t s = 0; s <= num_splits; ++s)
        {
            total_cost_each_split[s] += io_cost_each_split[s][q];
            if (io_cost_each_split[s][q] == best_io)
            {
                ++num_best_performance_each_split[s];
                ++total_for_likelihood;
            }
        }
    }

    split_scores.clear();
    std::vector< double > split_likelihoods;

    double min_overlap = geometric_stats[0];
    double min_area = geometric_stats[1];
    double min_margin = geometric_stats[2];
    double min_total_area = geometric_stats[3];
    double min_total_margin = geometric_stats[4];

    double max_overlap = geometric_stats[5];
    double max_area = geometric_stats[6];
    double max_margin = geometric_stats[7];
    double max_total_area = geometric_stats[8];
    double max_total_margin = geometric_stats[9];

    double avg_area = geometric_stats[10];
    double avg_margin = geometric_stats[11];
    double avg_total_area = geometric_stats[12];
    double avg_total_margin = geometric_stats[13];

    double avg_score = 0;

    double avg_likelihood = 0;
    double likelihood_sum = 0;

    if (total_best_cost > 0)
    {
        // if (print_info)
        // {
        //     std::cout << "Step-1:" << std::endl;
        // }
        for (uint64_t s = 0; s < num_splits; ++s)
        {
            SplitInfo &split_info = split_infos[s];

            double score = static_cast<double>(total_best_cost) / static_cast<double>(total_cost_each_split[s]);
            avg_score += score;

            double likelihood = static_cast<double>(num_best_performance_each_split[s]) / static_cast<double>(count);
            // likelihood = exp(likelihood * 5);
            avg_likelihood += likelihood;

            split_scores.emplace_back(score);
            split_likelihoods.emplace_back(likelihood);

            // if (print_info)
            // {
            //     std::cout << "\t" << "s = " << s << ", score = " << score << ", likelihood = " << likelihood << std::endl;
            // }
        }
    }

    SplitInfo &padding_split_info = split_infos[num_splits];
    if (total_best_cost > 0)
    {
        double score = static_cast<double>(total_best_cost) / static_cast<double>(total_cost_each_split[num_splits]);
        split_scores.emplace_back(score);
        split_likelihoods.emplace_back(0);
        avg_likelihood /= static_cast<double>(num_splits);
        avg_score /= static_cast<double>(num_splits);
    }

    std::vector< double > split_geometric_scores;
    double area_weight = 0.9;
    double margin_weight = 0.1;
    double overlap_weight = 0;

    double avg_geometric_score = 0;

    if (min_overlap == max_overlap)
    {
        overlap_weight = 0;
        margin_weight = 0.1;
        area_weight = 0.9;
    }

    // if (print_info)
    // {
    //     std::cout << "Step-2:" << std::endl;
    // }
    for (uint64_t s = 0; s < num_splits; ++s)
    {
        SplitInfo &split_info = split_infos[s];
        double overlap_score = 1;
        if (overlap_weight > 0 && max_overlap > 0)
        // if (overlap_weight > 0)
        {
            overlap_score = 1.0 - split_info.overlap / max_overlap;
        }
        double area_score = 1;
        double margin_score = 1;
        if (split_info.area > 0)
        {
            area_score = min_total_area / split_info.area;
        }
        if (split_info.margin > 0)
        {
            margin_score = min_total_margin / split_info.margin;
        }

        double score = overlap_score * overlap_weight +
            area_score * area_weight + margin_score * margin_weight;

        avg_geometric_score += score;
        split_geometric_scores.emplace_back(score);

        // if (print_info)
        // {
        //     std::cout << "\t" << "s = " << s << ", overlap = " << split_info.overlap << ", overlap_score = " << overlap_score << ", area = " << split_info.area << ", area_score = " << area_score
        //     << ", margin = " << split_info.margin << ", margin_score = " << margin_score << ", score = " << score << std::endl;
        // }
    }
    double geometric_score_sum = avg_geometric_score;
    avg_geometric_score /= static_cast<double>(num_splits);

    double overlap_score = 1;
    if (overlap_weight > 0)
    {
        overlap_score = -1;
    }
    double area_score = min_total_area / padding_split_info.area;
    double margin_score = min_total_margin / padding_split_info.margin;

    double geometric_score = overlap_score * overlap_weight +
        area_score * area_weight + margin_score * margin_weight;
    split_geometric_scores.emplace_back(geometric_score);

    if (total_best_cost == 0)
    {
        split_likelihoods = split_geometric_scores;
        split_scores = std::move(split_geometric_scores);
        likelihood_sum = geometric_score_sum;
        avg_score = avg_geometric_score;
    }
    else
    {
        double score_sum = 0;
        double geometric_weight = 0.05;
        // double io_weight = 1;
        likelihood_sum = 0;
        // if (print_info)
        // {
        //     std::cout << "Step-3:" << std::endl;
        // }
        for (uint64_t s = 0; s < num_splits; ++s)
        {
            split_scores[s] += split_geometric_scores[s] * geometric_weight * avg_score;
            split_likelihoods[s] += split_geometric_scores[s] * geometric_weight * avg_likelihood;
            // split_scores[s] = split_scores[s] * io_weight + split_geometric_scores[s] * geometric_weight * avg_score;
            // split_likelihoods[s] = split_likelihoods[s] * io_weight + split_geometric_scores[s] * geometric_weight * avg_likelihood;
            score_sum += split_scores[s];
            likelihood_sum += split_likelihoods[s];
            if (print_info)
            {
                std::cout << "\t" << "s = " << s << ", score = " << split_scores[s] << ", score_sum = " << score_sum << ", likelihood_sum = " << likelihood_sum << std::endl;
            }
        }
        avg_score = score_sum / static_cast<double>(num_splits);
        // if (print_info)
        // {
        //     std::cout << "\t" << "avg_score = " << avg_score << std::endl;
        // }
    }

    int64_t chosen = -1;
    double max_score = 0;

    if (split_likelihood_as_score)
    {

        for (uint64_t s = 0; s <= num_splits; ++s)
        {
            double likelihood = split_likelihoods[s] / likelihood_sum;
            if (likelihood > max_score)
            {
                chosen = s;
                max_score = likelihood;
            }
            split_likelihoods[s] = likelihood;
        }
        split_scores = std::move(split_likelihoods);
    }
    else
    {
        double ratio = avg_score / 0.5;

        for (uint64_t s = 0; s <= num_splits; ++s)
        {
            double score = split_scores[s];
            score /= ratio;
            if (score > max_score)
            {
                chosen = s;
                max_score = score;
            }
            split_scores[s] = score;
        }
    }


    return chosen;
}


void Node::chooseBestSplit(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id, std::stack<id_type>& _pathBuffer, std::vector<uint32_t>& group1, std::vector<uint32_t>& group2)
{
    uint32_t root_level = m_pTree->getRootLevel();
    std::stack<id_type> _insert_path(_pathBuffer);
    std::vector<id_type> insert_path;
    while (!_insert_path.empty())
    {
        insert_path.emplace_back(_insert_path.top());
        _insert_path.pop();
    }

    if ((insert_path.size() + m_level) != root_level)
    {
        uint64_t left = insert_path.size() + m_level;
        uint64_t right = m_pTree->getRootLevel();
        throw Tools::IsNotEqualException<uint64_t>("Node::chooseBestSplit", "insert_path.size() + m_level", "m_pTree->getRootLevel()", left, right);
    }

    std::vector< SplitInfo > split_infos;

    Region union_bb = m_nodeMBR;
    union_bb.combineRegion(mbr);

    RTree *queryTree = m_pTree->queryTree;
    std::vector<Region> queries;
    queryTree->getOverlapMBRs(union_bb, queries);
    if (queries.empty())
    {
        queries.emplace_back(mbr);
    }
    enumerateCandidateSplit(dataLength, pData, mbr, id, split_infos);

    split_infos.emplace_back();
    SplitInfo &padding_split_info = split_infos.back();
    get_padding_split_info(union_bb, padding_split_info);

    std::vector< std::vector<uint64_t> > io_cost_each_split(split_infos.size());
    std::vector<double> cost_each_split;

    uint64_t max_cost = 0;
    for (uint32_t k = 0; k < split_infos.size(); ++k)
    {
        uint64_t cost_i = calc_split_cost_w_queries(insert_path, split_infos, k, queries, io_cost_each_split[k]);
        if (cost_i > max_cost)
        {
            max_cost = cost_i;
        }
    }

    std::vector<double> split_scores;

    std::vector< double > geometric_stats;
    get_split_geometric_properties(split_infos, geometric_stats);

    int64_t best_index = get_split_scores(split_infos, geometric_stats, io_cost_each_split, split_scores, m_pTree->split_likelihood_as_score);

    bool flag = (best_index >= 0);
    if (!flag)
    {
        throw Tools::BasicException("In Node::chooseBestSplit, flag should be true. Now it is false. num_queries = " + std::to_string(queries.size()));
    }

    SplitInfo &split_info = split_infos[best_index];
    group1.clear();
    group2.clear();
    group1 = std::move(split_info.group1);
    group2 = std::move(split_info.group2);
    if((group1.size() + group2.size()) != m_capacity + 1)
    {
        uint64_t left = group1.size() + group2.size();
        uint64_t right = m_capacity + 1;
        throw Tools::IsNotEqualException<uint64_t>("Node::chooseBestSplit", "g1.size() + g2.size()", "m_capacity + 1", left, right);
    }

    if (isSplitRecordingTurnedOn())
    {
        uint32_t cIndex;

        std::vector<Region> child_mbrs;
        for (cIndex = 0; cIndex <= m_capacity; ++cIndex)
        {
            child_mbrs.emplace_back(*(m_ptrMBR[cIndex]));
        }

        bool isLeaf = true;
        if (m_level > 0)
        {
            isLeaf = false;
        }
        m_pTree->writeASplitRecord(union_bb, child_mbrs, split_infos, split_scores, isLeaf);
    }
}

void Node::chooseSplit_w_model(uint32_t dataLength, uint8_t* pData, Region& mbr, id_type id, std::stack<id_type>& _pathBuffer, std::vector<uint32_t>& group1, std::vector<uint32_t>& group2)
{
    // bool print_info = m_pTree->split_debug;
    uint32_t root_level = m_pTree->getRootLevel();
    std::stack<id_type> _insert_path(_pathBuffer);
    std::vector<id_type> insert_path;
    while (!_insert_path.empty())
    {
        insert_path.emplace_back(_insert_path.top());
        _insert_path.pop();
    }

    // if (print_info)
    // {
    // std::cout << "0. insert_path.size() = " << insert_path.size() << ", m_level = " << m_level << std::endl;
    // }
    if ((insert_path.size() + m_level) != root_level)
    {
        uint64_t left = insert_path.size() + m_level;
        uint64_t right = m_pTree->getRootLevel();
        throw Tools::IsNotEqualException<uint64_t>("Node::chooseBestSplit", "insert_path.size() + m_level", "m_pTree->getRootLevel()", left, right);
    }

    // if ((insert_path.size() + m_level) != root_level)
    // {
    //     uint64_t left = insert_path.size() + m_level;
    //     uint64_t right = m_pTree->getRootLevel();
    //     throw Tools::IsNotEqualException<uint64_t>("Node::chooseBestSplit", "insert_path.size() + m_level", "m_pTree->getRootLevel()", left, right);
    // }

    std::vector< SplitInfo > split_infos;
    Region union_bb = m_nodeMBR;
    union_bb.combineRegion(mbr);

    enumerateCandidateSplit(dataLength, pData, mbr, id, split_infos);

    if (split_infos.size() > (m_pTree->split_model_config_).max_num_splits)
    {
        std::sort(split_infos.begin(), split_infos.end(), SplitInfo::compare);
        split_infos.resize((m_pTree->split_model_config_).max_num_splits);
    }
    else
    {
        split_infos.emplace_back();
        SplitInfo &padding_split_info = split_infos.back();
        get_padding_split_info(union_bb, padding_split_info);
    }
    // std::sort(split_infos.begin(), split_infos.end(), SplitInfo::compare);
    // uint64_t max_size = 10;
    // if (max_size > split_infos.size())
    // {
    //     max_size = split_infos.size();
    // }
    // split_infos.resize(max_size);

    // if (print_info)
    // {
    // std::cout << "1. split_infos.size() = " << split_infos.size() << std::endl;
    // }

    uint32_t cIndex;
    std::vector<Region> child_mbrs;
    for (cIndex = 0; cIndex <= m_capacity; ++cIndex)
    {
        child_mbrs.emplace_back(*(m_ptrMBR[cIndex]));
    }
    bool isLeaf = true;
    if (m_level > 0)
    {
        isLeaf = false;
    }

    std::vector< double > geometric_stats;
    get_split_geometric_properties(split_infos, geometric_stats);
    int64_t pred_index = m_pTree->splitPredict(union_bb, child_mbrs, split_infos, isLeaf);

    // if (print_info)
    // {
    // std::cout << "2. pred_index = " << pred_index << std::endl;
    // }

    bool flag = (pred_index >= 0);
    if (!flag)
    {
        throw Tools::BasicException("In Node::chooseSplit_w_model, flag should be true. Now it is false.");
    }


    if (isSplitRecordingTurnedOn())
    {
        RTree *queryTree = m_pTree->queryTree;
        std::vector<Region> queries;
        queryTree->getOverlapMBRs(union_bb, queries);

        std::vector< std::vector<uint64_t> > io_cost_each_split(split_infos.size());
        std::vector<double> cost_each_split;

        for (uint32_t k = 0; k < split_infos.size(); ++k)
        {
            calc_split_cost_w_queries(insert_path, split_infos, k, queries, io_cost_each_split[k]);
        }

        std::vector<double> split_scores;
        int64_t best_index = get_split_scores(split_infos, geometric_stats, io_cost_each_split, split_scores, m_pTree->split_likelihood_as_score);

        flag = (best_index >= 0);
        if (!flag)
        {
            throw Tools::BasicException("In Node::chooseSplit_w_model, the second flag should be true. Now it is false.");
        }
        m_pTree->writeASplitRecord(union_bb, child_mbrs, split_infos, split_scores, isLeaf);
    }

    SplitInfo &split_info = split_infos[pred_index];
    group1.clear();
    group2.clear();
    group1 = std::move(split_info.group1);
    group2 = std::move(split_info.group2);
    // std::cout << "Terminate chooseBestSplit()." << std::endl;

    if((group1.size() + group2.size()) != m_capacity + 1)
    {
        uint64_t left = group1.size() + group2.size();
        uint64_t right = m_capacity + 1;
        throw Tools::IsNotEqualException<uint64_t>("Node::chooseSplit_w_model", "g1.size() + g2.size()", "m_capacity + 1", left, right);
    }
}



void Node::pickSeeds_w_rv(uint32_t& index1, uint32_t& index2, RTreeVariant rv)
{
	double separation = -std::numeric_limits<double>::max();
	double inefficiency = -std::numeric_limits<double>::max();
	uint32_t cDim, u32Child, cIndex;

	switch (rv)
	{
		case RV_LINEAR:
		case RV_RSTAR:
	    // case RV_BEST_SUBTREE:
	    // case RV_MODEL_SUBTREE
	    // case RV_BEST_SPLIT:
	    // case RV_MODEL_SPLIT:
	    // case RV_BEST_BOTH:
	    // case RV_MODEL_BOTH:
			for (cDim = 0; cDim < m_pTree->m_dimension; ++cDim)
			{
				double leastLower = m_ptrMBR[0]->m_pLow[cDim];
				double greatestUpper = m_ptrMBR[0]->m_pHigh[cDim];
				uint32_t greatestLower = 0;
				uint32_t leastUpper = 0;
				double width;

				for (u32Child = 1; u32Child <= m_capacity; ++u32Child)
				{
					if (m_ptrMBR[u32Child]->m_pLow[cDim] > m_ptrMBR[greatestLower]->m_pLow[cDim]) greatestLower = u32Child;
					if (m_ptrMBR[u32Child]->m_pHigh[cDim] < m_ptrMBR[leastUpper]->m_pHigh[cDim]) leastUpper = u32Child;

					leastLower = std::min(m_ptrMBR[u32Child]->m_pLow[cDim], leastLower);
					greatestUpper = std::max(m_ptrMBR[u32Child]->m_pHigh[cDim], greatestUpper);
				}

				width = greatestUpper - leastLower;
				if (width <= 0) width = 1;

				double f = (m_ptrMBR[greatestLower]->m_pLow[cDim] - m_ptrMBR[leastUpper]->m_pHigh[cDim]) / width;

				if (f > separation)
				{
					index1 = leastUpper;
					index2 = greatestLower;
					separation = f;
				}
			}  // for (cDim)

			if (index1 == index2)
			{
				if (index2 == 0) ++index2;
				else --index2;
			}

			break;
		case RV_QUADRATIC:
		case RV_BEST_SUBTREE:
	    case RV_MODEL_SUBTREE:
        case RV_BEST_SPLIT:
	    case RV_MODEL_SPLIT:
	    case RV_MODEL_BOTH:
			// for each pair of Regions (account for overflow Region too!)
			for (u32Child = 0; u32Child < m_capacity; ++u32Child)
			{
				double a = m_ptrMBR[u32Child]->getArea();

				for (cIndex = u32Child + 1; cIndex <= m_capacity; ++cIndex)
				{
					// get the combined MBR of those two entries.
					Region r;
					m_ptrMBR[u32Child]->getCombinedRegion(r, *(m_ptrMBR[cIndex]));

					// find the inefficiency of grouping these entries together.
					double d = r.getArea() - a - m_ptrMBR[cIndex]->getArea();

					if (d > inefficiency)
					{
						inefficiency = d;
						index1 = u32Child;
						index2 = cIndex;
					}
				}  // for (cIndex)
			} // for (u32Child)

			break;
		default:
		    throw Tools::NotSupportedException("Node::pickSeeds: Tree variant not supported.");
	}
}

void Node::pickSeeds(uint32_t& index1, uint32_t& index2)
{
	double separation = -std::numeric_limits<double>::max();
	double inefficiency = -std::numeric_limits<double>::max();
	uint32_t cDim, u32Child, cIndex;

	switch (m_pTree->m_treeVariant)
	{
		case RV_LINEAR:
		case RV_RSTAR:
	    // case RV_BEST_SUBTREE:
	    // case RV_MODEL_SUBTREE:
	    // case RV_BEST_SPLIT:
	    // case RV_MODEL_SPLIT:
	    // case RV_BEST_BOTH:
	    // case RV_MODEL_BOTH:
			for (cDim = 0; cDim < m_pTree->m_dimension; ++cDim)
			{
				double leastLower = m_ptrMBR[0]->m_pLow[cDim];
				double greatestUpper = m_ptrMBR[0]->m_pHigh[cDim];
				uint32_t greatestLower = 0;
				uint32_t leastUpper = 0;
				double width;

				for (u32Child = 1; u32Child <= m_capacity; ++u32Child)
				{
					if (m_ptrMBR[u32Child]->m_pLow[cDim] > m_ptrMBR[greatestLower]->m_pLow[cDim]) greatestLower = u32Child;
					if (m_ptrMBR[u32Child]->m_pHigh[cDim] < m_ptrMBR[leastUpper]->m_pHigh[cDim]) leastUpper = u32Child;

					leastLower = std::min(m_ptrMBR[u32Child]->m_pLow[cDim], leastLower);
					greatestUpper = std::max(m_ptrMBR[u32Child]->m_pHigh[cDim], greatestUpper);
				}

				width = greatestUpper - leastLower;
				if (width <= 0) width = 1;

				double f = (m_ptrMBR[greatestLower]->m_pLow[cDim] - m_ptrMBR[leastUpper]->m_pHigh[cDim]) / width;

				if (f > separation)
				{
					index1 = leastUpper;
					index2 = greatestLower;
					separation = f;
				}
			}  // for (cDim)

			if (index1 == index2)
			{
				if (index2 == 0) ++index2;
				else --index2;
			}

			break;
		case RV_QUADRATIC:
		case RV_BEST_SUBTREE:
	    case RV_MODEL_SUBTREE:
        case RV_BEST_SPLIT:
	    case RV_MODEL_SPLIT:
	    case RV_MODEL_BOTH:
			// for each pair of Regions (account for overflow Region too!)
			for (u32Child = 0; u32Child < m_capacity; ++u32Child)
			{
				double a = m_ptrMBR[u32Child]->getArea();

				for (cIndex = u32Child + 1; cIndex <= m_capacity; ++cIndex)
				{
					// get the combined MBR of those two entries.
					Region r;
					m_ptrMBR[u32Child]->getCombinedRegion(r, *(m_ptrMBR[cIndex]));

					// find the inefficiency of grouping these entries together.
					double d = r.getArea() - a - m_ptrMBR[cIndex]->getArea();

					if (d > inefficiency)
					{
						inefficiency = d;
						index1 = u32Child;
						index2 = cIndex;
					}
				}  // for (cIndex)
			} // for (u32Child)

			break;
		default:
		    throw Tools::NotSupportedException("Node::pickSeeds: Tree variant not supported.");
	}
}

void Node::condenseTree(std::stack<NodePtr>& toReinsert, std::stack<id_type>& pathBuffer, NodePtr& ptrThis)
{
	uint32_t minimumLoad = static_cast<uint32_t>(std::floor(m_capacity * m_pTree->m_fillFactor));

	if (pathBuffer.empty())
	{
		// eliminate root if it has only one child.
		if (m_level != 0 && m_children == 1)
		{
			NodePtr ptrN = m_pTree->readNode(m_pIdentifier[0]);
			m_pTree->deleteNode(ptrN.get());
			ptrN->m_identifier = m_pTree->m_rootID;
			m_pTree->writeNode(ptrN.get());

			m_pTree->m_stats.m_nodesInLevel.pop_back();
			m_pTree->m_stats.m_u32TreeHeight -= 1;
			// HACK: pending deleteNode for deleted child will decrease nodesInLevel, later on.
			m_pTree->m_stats.m_nodesInLevel[m_pTree->m_stats.m_u32TreeHeight - 1] = 2;
		}
		else
		{
			// due to data removal.
			if (m_pTree->m_bTightMBRs)
			{
				for (uint32_t cDim = 0; cDim < m_nodeMBR.m_dimension; ++cDim)
				{
					m_nodeMBR.m_pLow[cDim] = std::numeric_limits<double>::max();
					m_nodeMBR.m_pHigh[cDim] = -std::numeric_limits<double>::max();

					for (uint32_t u32Child = 0; u32Child < m_children; ++u32Child)
					{
						m_nodeMBR.m_pLow[cDim] = std::min(m_nodeMBR.m_pLow[cDim], m_ptrMBR[u32Child]->m_pLow[cDim]);
						m_nodeMBR.m_pHigh[cDim] = std::max(m_nodeMBR.m_pHigh[cDim], m_ptrMBR[u32Child]->m_pHigh[cDim]);
					}
				}
			}

            // write parent node back to storage.
			m_pTree->writeNode(this);
		}
	}
	else
	{
		id_type cParent = pathBuffer.top(); pathBuffer.pop();
		NodePtr ptrParent = m_pTree->readNode(cParent);
		Index* p = static_cast<Index*>(ptrParent.get());

		// find the entry in the parent, that points to this node.
		uint32_t child;

		for (child = 0; child != p->m_children; ++child)
		{
			if (p->m_pIdentifier[child] == m_identifier) break;
		}

		if (m_children < minimumLoad)
		{
			// used space less than the minimum
			// 1. eliminate node entry from the parent. deleteEntry will fix the parent's MBR.
			p->deleteEntry(child);
			// 2. add this node to the stack in order to reinsert its entries.
			toReinsert.push(ptrThis);
		}
		else
		{
			// adjust the entry in 'p' to contain the new bounding region of this node.
			*(p->m_ptrMBR[child]) = m_nodeMBR;

			// global recalculation necessary since the MBR can only shrink in size,
			// due to data removal.
			if (m_pTree->m_bTightMBRs)
			{
				for (uint32_t cDim = 0; cDim < p->m_nodeMBR.m_dimension; ++cDim)
				{
					p->m_nodeMBR.m_pLow[cDim] = std::numeric_limits<double>::max();
					p->m_nodeMBR.m_pHigh[cDim] = -std::numeric_limits<double>::max();

					for (uint32_t u32Child = 0; u32Child < p->m_children; ++u32Child)
					{
						p->m_nodeMBR.m_pLow[cDim] = std::min(p->m_nodeMBR.m_pLow[cDim], p->m_ptrMBR[u32Child]->m_pLow[cDim]);
						p->m_nodeMBR.m_pHigh[cDim] = std::max(p->m_nodeMBR.m_pHigh[cDim], p->m_ptrMBR[u32Child]->m_pHigh[cDim]);
					}
				}
			}
		}

		// write parent node back to storage.
		m_pTree->writeNode(p);

		p->condenseTree(toReinsert, pathBuffer, ptrParent);
	}
}
