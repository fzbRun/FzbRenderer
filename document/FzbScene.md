对于读取sceneXML的流程

1\.	scene会从sceneXML中读取全局信息 

2\.	scene初始化

&nbsp;	2.1 为scene添加各种全局信息

&nbsp;	  2.2	 创建mesh的描述符集合

&nbsp;	2.3 从sceneXML中读取数据

&nbsp;		2.3.1 读取相机信息

&nbsp;		2.3.2 读取material

&nbsp;		2.3.3 根据material的type创建shader，并创建shaderVariant

&nbsp;			2.3.3.1 material的属性类型，关闭没有传入的属性的宏

&nbsp;		    2.3.3.2 根据传入的vertexFormat参数和开启的宏，决定shaderVariant的vertexFormat，并与material同步。

&nbsp;		    2.3.4 读取mehs信息（包括光源）

&nbsp;			2.3.4.1 读取mesh的路径，从.obj中读取数据（顶点和索引），读取顶点时读入vertexFormat，其与material同步。

&nbsp;			2.3.4.2 识别mesh的materialID，将之与之前读取的mateiral关联

&nbsp;			2.3.4.3 根据mesh的vertexFormat将其放入集合中，用于后续压缩数据

&nbsp;	2.4 创建顶点和索引缓冲

&nbsp;		2.4.1 每个vertexFormat集合分别压缩，即相同vertexFormat的mesh的顶点数据进行压缩

&nbsp;			2.4.1.1 从mesh中读取顶点和索引数据

&nbsp;			2.4.1.2 根据当前已读的相同vertexFormat的mesh的索引数量和所有之前读取的（不同vertexFormat）的索引数量，为mesh的indexArrayOffset进行赋值，为后续渲染offset准备

&nbsp;			2.4.1.3 创建一个mesh的索引数组副本，根据已读的相同vertexFormat的mesh的索引数量，为该副本的索引值进行偏移

&nbsp;			2.4.1.4 将该vertexFormat的mesh的所有顶点数据和偏移后的索引副本数组放入两个大的压缩数组中

&nbsp;		2.4.2 将压缩数组传给一个压缩函数，进行压缩

&nbsp;		2.4.3 将压缩后的顶点数组和索引数组加入两个大的场景顶点和索引数组中

&nbsp;			2.4.3.1 如果之前的vertexFormat的顶点数组的float数不能整除当前vertexFormat，则需要加入一些padding

&nbsp;			      2.4.3.2 将整除后的结果加到索引压缩数组的索引值上去

&nbsp;		2.4.4 最后根据场景顶点数组和索引数组，创建顶点缓冲和索引缓冲

&nbsp;	2.5 创建部分顶点数据的顶点和索引缓冲：根据外部传入的vertexFormat，分析创建只有pos的顶点和索引缓冲或有pos和normal的顶点和索引缓冲

&nbsp;	2.6 根据shader，为每个shaderVariant创建meshBatch

&nbsp;		2.6.1 根据mesh带有的material是否存在于shaderVariant的materials中，如果存在则将其加入shaderVariant的meshBatch的meshes中；反之跳过

&nbsp;	2.7 为scene中的资源创建缓冲区和描述符

&nbsp;		2.7.1 每个material会根据自身的属性创建material数值buffer

&nbsp;		    2.7.2 根据scene中的material的数值buffer数量、所需纹理数量和是否有相机或光源，创建描述符池

&nbsp;		2.7.3 每个mesh创建meshBuffer，主要就是变换矩阵

&nbsp;		2.7.4 每个shader创建描述符

&nbsp;			2.7.4.1 每个shaderVariant根据自身的属性创建描述符集合布局（与自身的materials保持一致）

&nbsp;			2.7.4.2 materials中的material创建描述符集合

&nbsp;	  2.8 为相机和光照创建描述符集合布局和描述符集合





