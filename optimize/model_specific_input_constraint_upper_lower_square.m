#!/usr/local/bin/WolframScript -script

Print["Reading Inputs"]
weights=Import["verdict/mnist/para/weights2.csv", "CSV"];
biases=Import["verdict/mnist/para/biases2.csv", "List"];

inNum=Dimensions[weights][[1]];
outNum=Dimensions[weights][[2]];

weights=Transpose[weights];

threshold=10^-1
weightsPruned=Chop[weights, threshold];
biasesPruned=Chop[biases, threshold];

input=Table[img[i], {i, inNum}];

Print["Forward Calculation"]
output=weights.input+biases;
outputPruned=weightsPruned.input+biasesPruned;

Print["Generating Constraints"]
length=upperBound-lowerBound

f[x_] = If[x > 0, x*upperBound, x*lowerBound];
boundCond:=upperBound>=lowerBound>=0
  
uppers=Table[u[i], {i, outNum * 2}];
For[k = 1, k <= outNum, k++,
  upper1=Total[Map[f, weights[[k]]]]+biases[[k]];
  uppers[[k]]=upper1<=1.046903;
  upper2=Total[Map[f, weightsPruned[[k]]]]+biasesPruned[[k]];
  uppers[[outNum + k]]=upper2<=1.046903;
]
upperCond=And@@uppers;

Print["Start Solving"];
sol=Maximize[{length, Rationalize[upperCond && boundCond, 10^-18]}, {upperBound, lowerBound}];

Print["Validate"];
newInCond=inCond/.sol[[2]];
outCond:=Or@@Map[Greater[#,1.046903]&, output];
outPrunedCond:=Or@@Map[Greater[#,1.046903]&, outputPruned];
FindInstance[newInCond && (outCond || outPrunedCond), input];

