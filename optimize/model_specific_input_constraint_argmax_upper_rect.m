#!/usr/local/bin/WolframScript -script

Print["Reading Inputs"]
weights=Import["verdict/mnist/para/weights3.csv", "CSV"];
biases=Import["verdict/mnist/para/biases3.csv", "List"];

inNum=Dimensions[weights][[1]];
outNum=Dimensions[weights][[2]];

weights=Transpose[weights];
bounds=Table[b[i], {i, inNum}];

threshold=10^-1
weightsPruned=Chop[weights, threshold];
biasesPruned=Chop[biases, threshold];

input=Table[img[i], {i, inNum}];

Print["Forward Calculation"]
output=weights.input+biases;
outputPruned=weightsPruned.input+biasesPruned;

(* newWeights=Transpose[Join[Transpose[weights], {biases}]];
newWeightsPruned=Transpose[Join[Transpose[weights], {biases}]]; *)
(* diff[i][j] contains output[j]-output[i]  *)
weightsDiff=Table[d[i][j], {i, outNum}, {j, outNum}];
For[i=1,i<=outNum,i++,For[j=1,j<=outNum,j++,weightsDiff[[i]][[j]]=weights[[j]]-weights[[i]]]];
weightsDiffPruned=Table[d[i][j], {i, outNum}, {j, outNum}];
For[i=1,i<=outNum,i++,For[j=1,j<=outNum,j++,weightsDiffPruned[[i]][[j]]=weightsPruned[[j]]-weightsPruned[[i]]]];

biasesDiff=Table[d[i][j], {i, outNum}, {j, outNum}];
For[i=1,i<=outNum,i++,For[j=1,j<=outNum,j++,biasesDiff[[i]][[j]]=biases[[j]]-biases[[i]]]];
biasesDiffPruned=Table[d[i][j], {i, outNum}, {j, outNum}];
For[i=1,i<=outNum,i++,For[j=1,j<=outNum,j++,biasesDiffPruned[[i]][[j]]=biasesPruned[[j]]-biasesPruned[[i]]]];

Print["Generating Constraints"]
product:=Apply[Times, bounds];

boundCond:=And@@Map[GreaterEqual[#,0]&, bounds];
inCond:=And@@MapThread[GreaterEqual[#1,#2,0]&, {bounds, input}];
  
f[x_, y_] = If[x > 0, x*y, 0];
uppers=Table[u[argmax][k], {argmax, outNum}, {k, outNum * 2}];
(* argmax being 0 indicates output[0] > output[1], i.e., output[1] - output[0] < 0,
i.e., diff[0][1] < 0, i.e., max[diff[0][1]] < 0 *)
For[argmax = 1, argmax <= outNum, argmax++,
  For[k = 1, k <= outNum, k++,
    If[k == argmax, Continue[]];
    upper1=Total[MapThread[f, {weightsDiff[[argmax]][[k]], bounds}]]+biasesDiff[[argmax]][[k]];
    uppers[[argmax]][[k]]=upper1<=0;
    upper2=Total[MapThread[f, {weightsDiffPruned[[argmax]][[k]], bounds}]]+biasesDiffPruned[[argmax]][[k]];
    uppers[[argmax]][[k+outNum]]=upper2<=0;
  ]
]
upperCond=And@@uppers;

Print["Start Solving"];
sol=Maximize[{product, upperCond && boundCond}, bounds];

Print["Validate"];
newInCond=inCond/.sol[[2]];
outCond:=Or@@Map[Greater[#,1.046903]&, output];
outPrunedCond:=Or@@Map[Greater[#,1.046903]&, outputPruned];
FindInstance[newInCond && outCond && outPrunedCond, input]

