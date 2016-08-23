#!/usr/local/bin/WolframScript -script

Print["Reading Inputs"]
weights=Import["verdict/mnist/para/weights2.csv", "CSV"];
biases=Import["verdict/mnist/para/biases2.csv", "List"];

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

Print["Generating Constraints"]
sum:=Total[bounds];
product:=Apply[Times, bounds];

f[x_, y_] = If[x > 0, x*y, 0];
boundCond:=And@@Map[GreaterEqual[#,0]&, bounds];
inCond:=And@@MapThread[GreaterEqual[#1,#2,0]&, {bounds, input}];
  
uppers=Table[u[i], {i, outNum * 2}];
For[k = 1, k <= outNum, k++,
  upper1=Total[MapThread[f, {weights[[k]], bounds}]]+biases[[k]];
  uppers[[k]]=upper1<=1.046903;
  upper2=Total[MapThread[f, {weightsPruned[[k]], bounds}]]+biasesPruned[[k]];
  uppers[[outNum + k]]=upper2<=1.046903;
]
upperCond=And@@uppers;

Print["Start Solving"];
sol=Maximize[{product, Rationalize[upperCond && boundCond, 10^-18]}, bounds];

Print["Validate"];
newInCond=inCond/.sol[[2]];
outCond:=Or@@Map[Greater[#,1.046903]&, output];
outPrunedCond:=Or@@Map[Greater[#,1.046903]&, outputPruned];
FindInstance[newInCond && (outCond || outPrunedCond), input]

