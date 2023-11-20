%% 02502 exam Spring 2022 solution

%% Aorta analysis
clc; clear; close all;

ct = dicomread('Aorta/1-442.dcm');
imshow(ct, [-100, 400]);

AortaROI = imread('Aorta/AortaROI.png');
AortaVals = double(ct(AortaROI));

meanAorta = mean(AortaVals);
stdAorta = std(AortaVals);

sprintf("Answer: average %.0f std: %.0f", meanAorta, stdAorta)


segm = ct > 90;
clearI = imclearborder(segm);
L8 = bwlabel(clearI,8);
%imshow(label2rgb(L8))
stats8 = regionprops(L8, 'Area', 'Perimeter');
circularity =  (4 * pi * [stats8.Area]) ./ ([stats8.Perimeter].^2);
idx = find([circularity] > 0.95  & [stats8.Area] > 200);
% pixel spacing in mm
ps = 0.75;
onepix =ps*ps;
all_area = [stats8.Area];
pix_area = all_area(idx(1));
% result 
cross_area = pix_area * onepix;
sprintf("Answer: Aorta area %.0f", cross_area)


LiverROI = imread('Aorta/LiverROI.png');
LiverVals = double(ct(LiverROI));
meanLiver = mean(LiverVals);
stdLiver = std(LiverVals);

xrange = -100:0.1:400;

pdfFitLiver = normpdf(xrange, mean(LiverVals), std(LiverVals));
pdfFitAorta = normpdf(xrange, mean(AortaVals), std(AortaVals));

% Plot fitted Gaussians
plot(xrange,pdfFitLiver, xrange, pdfFitAorta);
legend('Liver','Aorta');
xlim([147 153]);
% Answer: 151

%% Road analysis
clc; clear; close all;

road = rgb2hsv(imread("CarData/road.png")); 
V = road(:, :, 3);
im_bin = double(V > 0.9); 
L8 = bwlabel(im_bin, 8); 
stats8 = regionprops(L8, 'area'); 
allArea = [stats8.Area];

AreaSorted = sort(allArea); 
result = AreaSorted(end - 1); 
sprintf("Answer: The minimum area is %.0f", result)


idx = find(allArea >= result); 
BW = ismember(L8, idx); 
imshow(BW); 


%% Car analysis
clear; clc; close all;

im = imread('CarData/car.png');
HSV = rgb2hsv(im);
Scomp = HSV(:,:,2);
segm =  Scomp > 0.7;
se2 = strel('disk',6);
e1 = imerode(segm, se2);
se2 = strel('disk',4);
I = imdilate(e1, se2);
imshow(I);

% exam answer
result = sum(sum(I));
sprintf("Answer: foreground pixels %.0f", result)

%% Spoonz - The Perfect Spoon
clear; close all; clc;
W = 200;
H = 550;
imgs = zeros(H * W, 6); 
for i = 1:6
    imgs(:, i) = reshape(imread("ImagePCA/spoon" + string(i) + ".png"), [H * W, 1]); 
end 

meanVec = mean(imgs, 2);
meanI = reshape(meanVec, H, W);
%imshow(meanI,[]);
result = meanI(500, 100);
sprintf("Answer: average pixel value %.0f", result)

% PCA on thresholded images
thres_img = double(imgs > 100);
[Vecs, Vals, Psi] = pc_evectors(thres_img, 5);

% Normalize the eigenvalues to get percent explained variation
Valsnorm = Vals / sum(Vals) * 100;
figure;
plot(Valsnorm, '*-')
ylabel('Percent explained variance')
xlabel('Principal component')

% Answer
sprintf("Answer percent explained by first component: %.1f", Valsnorm(1))

% PCA on grey scale images
[Vecs, Vals, Psi] = pc_evectors(imgs, 6);

% Normalize the eigenvalues to get percent explained variation
Valsnorm = Vals / sum(Vals) * 100;
figure;
plot(Valsnorm, '*-')
ylabel('Percent explained variance')
xlabel('Principal component')

% Answer
sprintf("Percent explained by component 1 and 2: %.1f", ...
    Valsnorm(1) + Valsnorm(2))

newspoon = imread('ImagePCA/spoon1.png');
newMat = double(reshape(newspoon, [], 1));
newMat = newMat - meanVec;
Proj = Vecs(:, 1:4)' * newMat;

% Answer
sprintf("Answer: Coordinates in PCA spcace: %.0f %.0f", Proj(1), Proj(2))



%% Point transformations
clear; close all; clc;

th = 20;
R = [cosd(th),-sind(th);sind(th),cosd(th)];
s = 2; 
tr = [3.1,-3.3];
x = [10,10];
x_trans = (R*x') * s + tr'

sprintf("Answer: Transformed position: %.1f %.1f", x_trans(1), x_trans(2))


%% Linear discriminant analysis (LDA) on volcanoes
clear; close all; clc;

% Training examples
% Passive (class 1)
X1=[1.2,2.9,1.7,1.8,3.2,3.1]
Y1=[1.1,0.4,-2.7,-0.3,1.3,-0.9]

% Erupting (class 2)
X2=[0.5,1.4,2.7,2];
Y2=[1.7,-2.1,-0.8,0.5];

% Shape input
Input=[[X1,X2]',[Y1,Y2]']
% Make class labels of class 1 and 2
Target=[zeros(1,length(X1)),ones(1,length(X2))]'

figure(2), clf, hold on, grid on
scatter(X1,Y1,'or')
scatter(X2,Y2,'ob')
%Note that the training and test examples are overlapping i.e. very noisy.

% To find the W use the LDA.m function.
W = LDA(Input,Target)

% Calculate linear log-scores for training data as described in the matlab code
L = [ones(length(Input),1) Input] * W';

% Calculate class probabilities as described in the matlab code
P = exp(L) ./ repmat(sum(exp(L),2),[1 2])


%Plot which training points are classified correctly
%C1:Normal 
q=find(P(1:length(X1),1)>0.5)
P(:,1)
scatter(X1(q),Y1(q),'xr')

%C2:Volcano that are wrongly classified
q=find(P(length(X1)+1:length(P),2)<=0.5)
scatter(X2(q),Y2(q),'xb')
legend('C1','C2','P(x=C1,C2)')

% Select the wrongly segmented class 1 training examples:
result = P(q+length(X1),2)

%Answer --> [0.27,0.39]

sprintf("Answer: Two wrong classified with probabilities: %.2f %.2f", ...
    result(1), result(2))


%% PCA Analysis with soccer data
clear; close all; clc;
load PCAdata/soccer_data.txt;

% Data matrix
X = soccer_data';

[M,N] = size(X);
% M is number of features and N is number of observations


% Plot scatterplots of all variables:
%[~,ax]=plotmatrix(X');

% PCA computation:
data = X;

% subtract off the mean for each dimension
mn = mean(data,2);
data = data - repmat(mn,1,N);

% calculate the covariance matrix
Cx = 1 / (N-1) * (data * data');

% find the eigenvectors and eigenvalues
[PC, V] = eig(Cx);

% extract diagonal of matrix as vector
V = diag(V);

% sort the variances in decreasing order
[junk, rindices] = sort(-1*V);
V = V(rindices);
PC = PC(:,rindices);

% project the original data set
signals = PC' * data;

% Exam answer
answer = max(max(abs(signals)));
sprintf("Answer: Maximum projected absolute value: %.2f", ...
    answer(1))

% plot explained variance of principal components
plot(V)
Vnorm = V / sum(V) * 100
Vnorm_org = Vnorm
plot(Vnorm, '*-')
ylabel('Percent explained variance')
xlabel('Principal component')


%% Landmark based registration
clear, close all,clc;

play1 = imread('Landmarks/play1.png');
play5 = imread('Landmarks/play5.png');
ref   =  imread('Landmarks/reference.png');

load('Landmarks/playfixedPoints.mat');
load('Landmarks/playmovingPoints.mat');

p1 = fixedPoints(1, :)
p2 = movingPoints(1, :)
ex = p2-p1
dist = sqrt(ex * ex')

mytform = fitgeotrans(movingPoints, fixedPoints,'NonreflectiveSimilarity');
play5t = imwarp(play5, mytform);

t1 = play5t < 180;
res = dice(t1, ref);

sprintf("Answer: Distance between landmarks: %.1f", dist)
sprintf("Answer: DICE score: %.2f", res)


%% Grain quality classification
clc; clear; close all;

% N1(25,10^2) bad quality
m1=25
s1=10

% N2(52,2^2) medium quality
m2=52
s2=2

% N3(150,30^2) high quality
m3=150
s3=30

% First we visualize the distributions using normpdf (Lx 6.5).
figure(1), clf;
x=[45:0.1:160];
hold on
plot(x,normpdf(x,m2,s2),'b');
plot(x,normpdf(x,m3,s3),'r');

% Parametric threshold
s1a=s2;
s2a=s3;
m1a=m2;
m2a=m3;
%Find threshold and its two solutions p 135. 
th2pPar=(s1a^2*m2a-s2a^2*m1a+sqrt(-s1a^2*s2a^2*(2*m2a*m1a-m2a^2-2*s2a^2*log10(s2a/s1a)/log10(exp(1))-m1a^2+2*s1a^2*log10(s2a/s1a)/log10(exp(1)))))/(-s2a^2+s1a^2);
th2nPar=(s1a^2*m2a-s2a^2*m1a-sqrt(-s1a^2*s2a^2*(2*m2a*m1a-m2a^2-2*s2a^2*log10(s2a/s1a)/log10(exp(1))-m1a^2+2*s1a^2*log10(s2a/s1a)/log10(exp(1)))))/(-s2a^2+s1a^2);
plot([th2pPar,th2pPar],[0,0.1],'c');
plot([th2nPar,th2nPar],[0,0.1],'--c');
sprintf("Answer: Pixel value separating medium from high: %.1f", th2nPar)

% Minimum distance threshold
mindist = (m1 + m2) / 2;
sprintf("Answer: Pixel value separating bad from medium: %.1f", mindist)

