# JEAR
在做数学题目的题意理解中,有遇到个问题,涉及到实体类型以及实体关系的抽取,故而对ACL2017上的与其相关论文进行复现

## paper简介

[Going out on a limb: Joint Extraction of Entity Mentions and Relations without Dependency 
Trees](https://www.aclweb.org/anthology/P/P17/P17-1085.pdf)采用了两个loss（label，relation）相加同时训练
实体类型以及关系类型，在实体类型中，作者采用bi-direction LSTM进行序列识别，同时在decode中connections to y(t-1)。
在关系类型识别中，作者采用了pointer networks预测目前的token与前面所有token的关系类型，将与前面token的概率向量拓展到R维
表示R个关系类型。但感觉存在缺陷，在拓展到多关系类型时候，作者采用了阈值控制输出所有大于阈值的relation，感觉不太舒服
</br>
[Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme](https://arxiv.org/pdf/1706.05075.pdf)
作者提出了一种tagging scheme用于实体类型&关系提取，每个token识别为tag（实体中的单词位置，关系类型和关系角色），如S-CP-2表示单个实体，
在关系CP中的第二个位置，然后就转化为序列标注任务即可。但存在几个问题：1）没有识别实体类型；2）tag总是很大2\*4*R+1
</br>
[Global Normalization of Convolutional Neural Networks for Joint Entity and Relation Classification](https://arxiv.org/pdf/1707.07719.pdf)
作者对文本进行CNN获得全局信息后分别对相应部分输出实体类型以及关系类型，然后进行CRF计算预测序列的概率。但感觉也存在几点缺陷：
1）每次需要单独输入实体，这就需要先进行实体提取，而且每对实体都进行关系识别，对模型计算也比较大；2）CRF特征提取液需要进行训练
作者有提供了[源代码](http://cistern.cis.lmu.de)。


经过整理，决定实现Going out on a limb: Joint Extraction of Entity Mentions and Relations without Dependency 
Trees，对Global Normalization of Convolutional Neural Networks for Joint Entity and Relation Classification感兴趣的可以去看下作者提供的
[源代码](http://cistern.cis.lmu.de)