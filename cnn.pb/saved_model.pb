??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58??
v
ConstConst*
_output_shapes
:
*
dtype0*=
value4B2
"(?ܾ?V???E???`?sv?=?Q???U?=?????q?G?
?
Const_1Const*
_output_shapes

:
d*
dtype0*?
value?B?
d"?Kҙ??y??>?8>??A>??>և?=?m=??.>B??>???D7????$?M_?;[??;V?????I?-?H????
bU>^
(???????/??l???:
??8??7???ՙ?????>??s<?sĻ?;???<4?!=?h???1[?:?m?7??>P4?=(b?<KPM=U?=??>=?-^<?J??}??XW???'?=*??=T?z?c?<CAO>-Kd><?>a@Ǽan???㽧\?=#%C=?x???a?????<???=????]ν`???>y:??V>D?k?}e2???f??>?????jQ????????????<v?m>??.>'?I?K?Q??7M??gO?5?;?????????9??>??,A?>?d?=Q?y:?0??2mĻu?*=}?'<?yR?F?=Xa?w???{??@??pk???yk?I?k???b???J??V?8]?=?O>F?`>#?<>V?%>??>??>=C?=-=????m?=?:>?>!->??R>	a.>?%>?>?g?=
ὗfO???S?h?/?+???84J=Ǚ=?f?=o?(>MZ?= ?ν?#ɾ?qs?ZLm?Yu????2??;?q?=A7
>fw㼴Z??˗˾}?w??Z???`????\S=@)>^?=???@??X????p??_?W=?x?=-??=P0>/?j>?LA?FiӾ??"???????I?=?Y7>??Q>?n>_\>W9???C??)?c逾\?m?JR?=??<>V?D>6A>"p?=?،??k?9R'[>k?=??u?b?'<??&=@?e?lfͽ]?C?g;??Jp!?]??;L?>?C?=??=??,>C?>???=?/>??,>+;?=?)?????[߽???/? ?????4?3?????YżE?-<?FjB?????&&+?c??^w??%@?B!
?	I?????¾7?$<?i??bC}?S???񎽏]??~?0?~[?<I3=?Ǿ?#?=_)>1??=o??=W??=??='`û??);?@?=WQ?@>??H>?<>kR?=)}=z0	=???9䈽\G?=????N??<M=?<?C????`??^컽?Q????=d %?(0?բݽ??ý|IM???:&#??n1??2̽??=萼?????o[??h
?E?<???=??s? ??????k>??2>&&>?a5>?S/>b??=[4p???8_X?!0=???'??)?>p,>??<?1=???=???=???=?A>Us?=???;????,??I?]?????????-߽?1???t???????<??ƾ???$???^??}???ͽG7??݌r???w??V>b<?d?i="?>??=???-????RԻ}I?<I>tB??!?=?#/>M?=?BG=*=?'
=?l=_)E=?d?=?$??:?<\?>?`>?}?=???=J ?=???7(8??o?<>P???? ?֩?<5nb=???=??=?&?<???0g8??ŉ0??,ٽ?D????t???9e?-={tѻ?]s?o?q?5/???i?J?D?1???0g۽?9????.;|瘽G???ZJ={??=??;?kH??0?Ƚ? нx.?????<??="?}>V?B+ڽ?J??*??<???2????U?m6??RSѽ?F??=?Z=?4?=m.>B?'>qS#>T?>??=th>?2h>P????l?<?3=?=?5A>`?q>?v?>?n>?q*>?b?j?*=??<??H??n;?ORv<???=???=Qjv=?=? N?S}>ތ?<$?5F?????o????R?lp\?w*<!f???I> k?4??0?)/??У_??D?mg3????<ud?=w=?\?>??8=5|?????Fݽr"??À=???=sr?=]	g=z?>J??=;?T:R ?7?'=???=???=	`=?E<??iw>??ռ?㾽?H"?ܓ	>?)>??=?Ȧ<$?y?kq??n?Ӿg?输?ξ?6:?C??=Kg>mAQ>?S?=??˽??????>#s?=??<?Ē??????0??????F????=(?#?y??=?=?4?<??;????l??(>нo5q?%wC???%>A??<?פ?m?<??=`?P=,?<QUF?}??u!F?p??Ws?<=\?e??<N??=?*?=?
?=??<?*??s?+}??߁=?Z??3??=H=??>.>?t?=y}; F??a????>A????2??| ;B??=??=\I=?ά;?????Z???qT>3?.?&q?????d?????X??Ta??Y?=u ?=??;?-??=???\?????0??k?ɽ??)?^?=U??=?0z=?v?=??V?`??H??a?H????B?????<???=?`<???>'?>J??w?B< ??<??i??2ͽR	?4R???
^?ς?(*??8?־D̶?(m???"???Q~?oO??a???V?&????=D&N>?+`>??o>??>?P?=mV>?Q>??U?E?}???0>@!1>?1>? >k??=???=??*>N~>?w==f???<w=?V?<潈<?V????e=??=?Eu=??<Ͳ>????,Ľ𒟽?????ͽ Or?ߨ???h?F=qr>:y??ල????E?Ͻ???y???*?W?,?[?%:T?p0U>?e??0C<#?=Y=9?=?`?.b??+⏾P??aLd>????0??="?;>%qN>?	>5B???,i??.V?V?ȼ??s>?eǼ???=??2>7	%>R?<???z,?{??!@?<`g;>?]?=?=??>??=???<???:?b?=&?=?y?=?N?=\l?>??>?|??f??IR?Q?½`c="^?=4??=?iD=?????z??L?<?c?=?~?=??+>P?k>?7n>?Y'>?^=??h?3?"<??=攏=o^?<???<7t?=??=??R=q?ŽTm?/iq?(i?;???B[\????_????='?<S??V?????y?%?uPk???o?L.?!?Ͻ????=??Z?k??a:??rǽ	߽?u?V?ǽ??W ? V=K??={>??>???=??R=~=????/???7]1?mV=??r>??J>O?t>L_U>&#>+p?< x???6??h!?=
?->???>????#l?=???=?,?=?[	=S=럱=?<?=??=?~?<i????k(??B???6??6??=ĲV>?@>E????Rb??.?<???<?u?=.?=@?>951>?l?="𗼙????M>%>?d???*??I??(UϽ،??qF??`?Ľ?'0???p>3G$>?o??ڽ?u??$$??ViŽ???G?W?fA>????=5??=I?D???e?PRD<RO?<Ëa?6?9w??8H?Ͼ$=N?߼?p?:Fp?=?F>R?=7!3=????????????">??J?{??=6c/>?->?<?='??=??>!??=I??=?u$=????¢ =?)?=??Y=??< 3k=?,?=??
>N?=λӽ????l???BA:=* 5?X@???	???ֽ??z???I??|??>L???"?=bP>?? ?N????^?c>????1?=?O???eP?=???>?!?>
?=?f???"ڽU`
???= [;>?:??E???̬v?? ?=.m|>z?>^O?>?|>Cx?=?~	??\Ǿ?c>d?=$&{?V?~<???v??=l??=JU?:??.???>"?/<?>??#k??Pr?????*U?c?>ΐ*>8h?=?$B>??\?????????pQ?l?'??=??>??>Kk?=x??=?H???<<???<${ʼ??????[?j݀<v? =2ɔ?y??=2?0?.?߼?????߽??$?Dg???}?<??<???????=?!a???%?l???a.?-?b??)C???=?R=?	i???!>$??=?w;5ѷ;H
?<?$?=?b>?X>U@?=?????pX>R}?<.?=??<=.??=?C3>?U> "x>?.->?????(??.)Ӿx ži??
?D? 4??H?????<?DV?\)?
T
Const_2Const*
_output_shapes
:*
dtype0*
valueB*??I?
?
Const_3Const*&
_output_shapes
:*
dtype0*E
value<B:"$yf???????????2??O?>'?>??>?{?>
T
Const_4Const*
_output_shapes
:*
dtype0*
valueB*??a?
?
Const_5Const*&
_output_shapes
:*
dtype0*?
value|Bz"d.?n>?y????>?->>????>6???W ?>MP?U?=b???(?>p??>O??=|?> ??=v?	>?b?>e???ꏼ>ST?=X%>D?=ĨY>?? ?
T
Const_6Const*
_output_shapes
:*
dtype0*
valueB*0?:
?
Const_7Const*&
_output_shapes
:*
dtype0*?
value?B?"?&??>GZ??9勽l?=l?c???
>?s>C????:??X/>??2>?@???;>???]?=KlE<9َ>???=???>?">2F?>??\?I4>??>YS?>?>?(?<bh?>?2?=???>??#>?r>?/]>-?0>?}???_?>??+>?$?=???>b;>?$?=??>???>cH?=z??=????rF= !'??s?
T
Const_8Const*
_output_shapes
:*
dtype0*
valueB*?y??
?
Const_9Const*&
_output_shapes
:*
dtype0*?
value?B?"?m??ͻS?£?=y??Gǩ=????????9?>?>??>?I=o??>xX|<ĽҾ?J >m2???d???y????=ט??mv;?u???|>??>???=?m???<??
?OȽ?:>-??=??$>????cg>/TN>?DQ?׉f?X?K?S?ؽ??3;?5<??X=V#??ّ??9d???=Ɉ"?o8???7??
v
serving_default_inputPlaceholder*&
_output_shapes
:*
dtype0*
shape:
?
PartitionedCallPartitionedCallserving_default_inputConst_9Const_8Const_7Const_6Const_5Const_4Const_3Const_2Const_1Const*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference_signature_wrapper_713

NoOpNoOp
?
Const_10Const"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
}
handlers
outputs
initializer_dict
handler_variables
__call__
gen_tensor_dict

signatures* 

* 
* 
* 
* 

	trace_0* 
* 


serving_default* 
* 
?
	capture_0
	capture_1
	capture_2
	capture_3
	capture_4
	capture_5
	capture_6
	capture_7
	capture_8
	capture_9* 
?
	capture_0
	capture_1
	capture_2
	capture_3
	capture_4
	capture_5
	capture_6
	capture_7
	capture_8
	capture_9* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *%
f R
__inference__traced_save_839
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_restore_849??
?

?
!__inference_signature_wrapper_713	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity?
PartitionedCallPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *!
fR
__inference___call___686W
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?::::::::::
d:
:M I
&
_output_shapes
:

_user_specified_nameinput:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$	 

_output_shapes

:
d: 


_output_shapes
:

?
l
__inference__traced_save_839
file_prefix
savev2_const_10

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHo
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const_10"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
?J
?
__inference___call___806	
input
transpose_x	
add_y
transpose_3_x
add_1_y
transpose_6_x
add_2_y
transpose_9_x
add_3_y
transpose_12_x
mul_1_y
identityg
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             m
	transpose	Transposetranspose_xtranspose/perm:output:0*
T0*&
_output_shapes
:Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????y
splitSplitsplit/split_dim:output:0transpose:y:0*
T0*&
_output_shapes
:*
	num_spliti
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             k
transpose_1	Transposeinputtranspose_1/perm:output:0*
T0*&
_output_shapes
:?
convolutionConv2Dtranspose_1:y:0split:output:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
\
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????`
concat/concatIdentityconvolution:output:0*
T0*&
_output_shapes
:\
AddAddV2concat/concat:output:0add_y*
T0*&
_output_shapes
:i
transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             m
transpose_2	TransposeAdd:z:0transpose_2/perm:output:0*
T0*&
_output_shapes
:b
onnx_tf_prefix_/rel/ReluRelutranspose_2:y:0*
T0*&
_output_shapes
:i
transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             s
transpose_3	Transposetranspose_3_xtranspose_3/perm:output:0*
T0*&
_output_shapes
:\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
split_1Splitsplit_1/split_dim:output:0transpose_3:y:0*
T0*&
_output_shapes
:*
	num_spliti
transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
transpose_4	Transpose&onnx_tf_prefix_/rel/Relu:activations:0transpose_4/perm:output:0*
T0*&
_output_shapes
:?
convolution_1Conv2Dtranspose_4:y:0split_1:output:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
^
concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????d
concat_1/concatIdentityconvolution_1:output:0*
T0*&
_output_shapes
:b
Add_1AddV2concat_1/concat:output:0add_1_y*
T0*&
_output_shapes
:i
transpose_5/permConst*
_output_shapes
:*
dtype0*%
valueB"             o
transpose_5	Transpose	Add_1:z:0transpose_5/perm:output:0*
T0*&
_output_shapes
:d
onnx_tf_prefix_/rel_1/ReluRelutranspose_5:y:0*
T0*&
_output_shapes
:i
transpose_6/permConst*
_output_shapes
:*
dtype0*%
valueB"             s
transpose_6	Transposetranspose_6_xtranspose_6/perm:output:0*
T0*&
_output_shapes
:\
split_2/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
split_2Splitsplit_2/split_dim:output:0transpose_6:y:0*
T0*&
_output_shapes
:*
	num_spliti
transpose_7/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
transpose_7	Transpose(onnx_tf_prefix_/rel_1/Relu:activations:0transpose_7/perm:output:0*
T0*&
_output_shapes
:?
convolution_2Conv2Dtranspose_7:y:0split_2:output:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
^
concat_2/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????d
concat_2/concatIdentityconvolution_2:output:0*
T0*&
_output_shapes
:b
Add_2AddV2concat_2/concat:output:0add_2_y*
T0*&
_output_shapes
:i
transpose_8/permConst*
_output_shapes
:*
dtype0*%
valueB"             o
transpose_8	Transpose	Add_2:z:0transpose_8/perm:output:0*
T0*&
_output_shapes
:d
onnx_tf_prefix_/rel_2/ReluRelutranspose_8:y:0*
T0*&
_output_shapes
:i
transpose_9/permConst*
_output_shapes
:*
dtype0*%
valueB"             s
transpose_9	Transposetranspose_9_xtranspose_9/perm:output:0*
T0*&
_output_shapes
:\
split_3/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
split_3Splitsplit_3/split_dim:output:0transpose_9:y:0*
T0*&
_output_shapes
:*
	num_splitj
transpose_10/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
transpose_10	Transpose(onnx_tf_prefix_/rel_2/Relu:activations:0transpose_10/perm:output:0*
T0*&
_output_shapes
:?
convolution_3Conv2Dtranspose_10:y:0split_3:output:0*
T0*&
_output_shapes
:

*
paddingVALID*
strides
^
concat_3/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????d
concat_3/concatIdentityconvolution_3:output:0*
T0*&
_output_shapes
:

b
Add_3AddV2concat_3/concat:output:0add_3_y*
T0*&
_output_shapes
:

j
transpose_11/permConst*
_output_shapes
:*
dtype0*%
valueB"             q
transpose_11	Transpose	Add_3:z:0transpose_11/perm:output:0*
T0*&
_output_shapes
:

e
onnx_tf_prefix_/rel_3/ReluRelutranspose_11:y:0*
T0*&
_output_shapes
:

^
ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      
   
   ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: U
ProdProdstrided_slice:output:0Const:output:0*
T0*
_output_shapes
: F
SizeConst*
_output_shapes
: *
dtype0*
value	B :I
Const_1Const*
_output_shapes
: *
dtype0*
value	B :_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:\
strided_slice_1/stack_1PackSize:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stack_2PackConst_1:output:0*
N*
T0*
_output_shapes
:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:Q
Const_2Const*
_output_shapes
:*
dtype0*
valueB: [
Prod_1Prodstrided_slice_1:output:0Const_2:output:0*
T0*
_output_shapes
: c
Reshape/shapePackProd:output:0Prod_1:output:0*
N*
T0*
_output_shapes
:}
ReshapeReshape(onnx_tf_prefix_/rel_3/Relu:activations:0Reshape/shape:output:0*
T0*
_output_shapes

:d^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   m
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*
_output_shapes

:db
transpose_12/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_12	Transposetranspose_12_xtranspose_12/perm:output:0*
T0*
_output_shapes

:d
e
MatMulMatMulflatten/Reshape:output:0transpose_12:y:0*
T0*
_output_shapes

:
J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??U
mulMulmul/x:output:0MatMul:product:0*
T0*
_output_shapes

:
L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
mul_1Mulmul_1/x:output:0mul_1_y*
T0*
_output_shapes
:
K
add_4AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes

:
H
IdentityIdentity	add_4:z:0*
T0*
_output_shapes

:
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?::::::::::
d:
:M I
&
_output_shapes
:

_user_specified_nameinput:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$	 

_output_shapes

:
d: 


_output_shapes
:

?
E
__inference__traced_restore_849
file_prefix

identity_1??
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHr
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
2Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 X
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: J

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?J
?
__inference___call___686	
input
transpose_x	
add_y
transpose_3_x
add_1_y
transpose_6_x
add_2_y
transpose_9_x
add_3_y
transpose_12_x
mul_1_y
identityg
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             m
	transpose	Transposetranspose_xtranspose/perm:output:0*
T0*&
_output_shapes
:Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????y
splitSplitsplit/split_dim:output:0transpose:y:0*
T0*&
_output_shapes
:*
	num_spliti
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             k
transpose_1	Transposeinputtranspose_1/perm:output:0*
T0*&
_output_shapes
:?
convolutionConv2Dtranspose_1:y:0split:output:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
\
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????`
concat/concatIdentityconvolution:output:0*
T0*&
_output_shapes
:\
AddAddV2concat/concat:output:0add_y*
T0*&
_output_shapes
:i
transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             m
transpose_2	TransposeAdd:z:0transpose_2/perm:output:0*
T0*&
_output_shapes
:b
onnx_tf_prefix_/rel/ReluRelutranspose_2:y:0*
T0*&
_output_shapes
:i
transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             s
transpose_3	Transposetranspose_3_xtranspose_3/perm:output:0*
T0*&
_output_shapes
:\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
split_1Splitsplit_1/split_dim:output:0transpose_3:y:0*
T0*&
_output_shapes
:*
	num_spliti
transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
transpose_4	Transpose&onnx_tf_prefix_/rel/Relu:activations:0transpose_4/perm:output:0*
T0*&
_output_shapes
:?
convolution_1Conv2Dtranspose_4:y:0split_1:output:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
^
concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????d
concat_1/concatIdentityconvolution_1:output:0*
T0*&
_output_shapes
:b
Add_1AddV2concat_1/concat:output:0add_1_y*
T0*&
_output_shapes
:i
transpose_5/permConst*
_output_shapes
:*
dtype0*%
valueB"             o
transpose_5	Transpose	Add_1:z:0transpose_5/perm:output:0*
T0*&
_output_shapes
:d
onnx_tf_prefix_/rel_1/ReluRelutranspose_5:y:0*
T0*&
_output_shapes
:i
transpose_6/permConst*
_output_shapes
:*
dtype0*%
valueB"             s
transpose_6	Transposetranspose_6_xtranspose_6/perm:output:0*
T0*&
_output_shapes
:\
split_2/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
split_2Splitsplit_2/split_dim:output:0transpose_6:y:0*
T0*&
_output_shapes
:*
	num_spliti
transpose_7/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
transpose_7	Transpose(onnx_tf_prefix_/rel_1/Relu:activations:0transpose_7/perm:output:0*
T0*&
_output_shapes
:?
convolution_2Conv2Dtranspose_7:y:0split_2:output:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
^
concat_2/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????d
concat_2/concatIdentityconvolution_2:output:0*
T0*&
_output_shapes
:b
Add_2AddV2concat_2/concat:output:0add_2_y*
T0*&
_output_shapes
:i
transpose_8/permConst*
_output_shapes
:*
dtype0*%
valueB"             o
transpose_8	Transpose	Add_2:z:0transpose_8/perm:output:0*
T0*&
_output_shapes
:d
onnx_tf_prefix_/rel_2/ReluRelutranspose_8:y:0*
T0*&
_output_shapes
:i
transpose_9/permConst*
_output_shapes
:*
dtype0*%
valueB"             s
transpose_9	Transposetranspose_9_xtranspose_9/perm:output:0*
T0*&
_output_shapes
:\
split_3/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
split_3Splitsplit_3/split_dim:output:0transpose_9:y:0*
T0*&
_output_shapes
:*
	num_splitj
transpose_10/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
transpose_10	Transpose(onnx_tf_prefix_/rel_2/Relu:activations:0transpose_10/perm:output:0*
T0*&
_output_shapes
:?
convolution_3Conv2Dtranspose_10:y:0split_3:output:0*
T0*&
_output_shapes
:

*
paddingVALID*
strides
^
concat_3/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????d
concat_3/concatIdentityconvolution_3:output:0*
T0*&
_output_shapes
:

b
Add_3AddV2concat_3/concat:output:0add_3_y*
T0*&
_output_shapes
:

j
transpose_11/permConst*
_output_shapes
:*
dtype0*%
valueB"             q
transpose_11	Transpose	Add_3:z:0transpose_11/perm:output:0*
T0*&
_output_shapes
:

e
onnx_tf_prefix_/rel_3/ReluRelutranspose_11:y:0*
T0*&
_output_shapes
:

^
ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      
   
   ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: U
ProdProdstrided_slice:output:0Const:output:0*
T0*
_output_shapes
: F
SizeConst*
_output_shapes
: *
dtype0*
value	B :I
Const_1Const*
_output_shapes
: *
dtype0*
value	B :_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:\
strided_slice_1/stack_1PackSize:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stack_2PackConst_1:output:0*
N*
T0*
_output_shapes
:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:Q
Const_2Const*
_output_shapes
:*
dtype0*
valueB: [
Prod_1Prodstrided_slice_1:output:0Const_2:output:0*
T0*
_output_shapes
: c
Reshape/shapePackProd:output:0Prod_1:output:0*
N*
T0*
_output_shapes
:}
ReshapeReshape(onnx_tf_prefix_/rel_3/Relu:activations:0Reshape/shape:output:0*
T0*
_output_shapes

:d^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   m
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*
_output_shapes

:db
transpose_12/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_12	Transposetranspose_12_xtranspose_12/perm:output:0*
T0*
_output_shapes

:d
e
MatMulMatMulflatten/Reshape:output:0transpose_12:y:0*
T0*
_output_shapes

:
J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??U
mulMulmul/x:output:0MatMul:product:0*
T0*
_output_shapes

:
L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
mul_1Mulmul_1/x:output:0mul_1_y*
T0*
_output_shapes
:
K
add_4AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes

:
H
IdentityIdentity	add_4:z:0*
T0*
_output_shapes

:
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?::::::::::
d:
:M I
&
_output_shapes
:

_user_specified_nameinput:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$	 

_output_shapes

:
d: 


_output_shapes
:
"?
J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default
6
input-
serving_default_input:0)
output
PartitionedCall:0
tensorflow/serving/predict:?
?
handlers
outputs
initializer_dict
handler_variables
__call__
gen_tensor_dict

signatures"
_generic_user_object
$
"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
?
	trace_02?
__inference___call___806?
???
FullArgSpec
args?
jself
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z	trace_0
?2??
???
FullArgSpec!
args?
jself
j
input_dict
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,

serving_default"
signature_map
 "
trackable_dict_wrapper
?
	capture_0
	capture_1
	capture_2
	capture_3
	capture_4
	capture_5
	capture_6
	capture_7
	capture_8
	capture_9B?
__inference___call___806input"?
???
FullArgSpec
args?
jself
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z	capture_0z	capture_1z	capture_2z	capture_3z	capture_4z	capture_5z	capture_6z	capture_7z	capture_8z	capture_9
?
	capture_0
	capture_1
	capture_2
	capture_3
	capture_4
	capture_5
	capture_6
	capture_7
	capture_8
	capture_9B?
!__inference_signature_wrapper_713input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z	capture_0z	capture_1z	capture_2z	capture_3z	capture_4z	capture_5z	capture_6z	capture_7z	capture_8z	capture_9
!J	
Const_9jtf.TrackableConstant
!J	
Const_8jtf.TrackableConstant
!J	
Const_7jtf.TrackableConstant
!J	
Const_6jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant?
__inference___call___806l
6?3
? 
,?)
'
input?
input"&?#
!
output?
output
?
!__inference_signature_wrapper_713l
6?3
? 
,?)
'
input?
input"&?#
!
output?
output
