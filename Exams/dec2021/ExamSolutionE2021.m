%% DTU Course 02502 exam solution E2021


%% Minimum distance and parametric classifier class ranges
clear,close all,clc

% Classes 1: grass, 2: road, 3: sky
val1 = [68, 65, 67]
val2 = [70, 80, 75]
val3 = [77, 92, 89]

m1 = mean(val1)
m2 = mean(val2)
m3 = mean(val3)

% min dist
t1 = (m1+m2) / 2
t2 = (m2+m3) / 2

s1 = std(val1)
s2 = std(val2)
s3 = std(val3)

% Parametric: Do some checks around the splitting point
v = 70
pgrass = normpdf(v, m1, s1)
proad = normpdf(v, m2, s2)
pgrass > proad

v = 81
pgrass = normpdf(v, m1, s1)
proad = normpdf(v, m2, s2)
psky =  normpdf(v, m3, s3)
proad > psky

figure;
xrange = 0:0.1:255; 
pdfFit1 = normpdf(xrange, m1, s1);
pdfFit2 = normpdf(xrange, m2, s2);
pdfFit3 = normpdf(xrange, m3, s3);

hold on;
plot(xrange, pdfFit1, 'r');
plot(xrange, pdfFit2, 'g');
plot(xrange, pdfFit3, 'b');
hold off;

xlim([60, 95]);


%% Color thresholding and DICE
clear; clc; close all;
im = imread('BikeImage/bikes.png');
%figure;
%imshow(im);

Rcomp = im(:,:,1);
Gcomp = im(:,:,2);
Bcomp = im(:,:,3);

%imshow(Gcomp);

% RGB segmentation
segm =  Gcomp > 200 & Rcomp < 100 & Bcomp > 100;
%figure;
%imshow(segm);

se2 = strel('disk',8);
e1 = imclose(segm, se2);

%figure;
imshow(e1);

GT_roi = imread('BikeImage/boxROI.png');

% Exam answer
res_dice = dice(e1, GT_roi)


%% Camera exercise - Ball
clear,close all,clc

% Ball diameter 20 cm
G = 0.20

% Distance to camera 150 cm
g = 1.50

% Chip size (m)
cw = 0.0054
ch = 0.0042

cwp = cw * 1200 * 1000
cwh = ch * 1200 * 1000

% Pixel width (m)
pw = cw/ cwp

f = 0.015

b = f;

% Ball length on CCD (in m)
B = b * G / g;

% Ball length in pixels
pix = B / pw

int32(pix)


%% point based registration
clear, close all,clc;

donald1 = imread('Donald/donald_1.png');
donald2 = imread('Donald/donald_2.png');

load('Donald/donaldfixedPoints.mat');
load('Donald/donaldmovingPoints.mat');

%figure;
%plot(fixedPoints(:,1), fixedPoints(:,2), 'b*-', ...
%movingPoints(:,1), movingPoints(:,2), 'r*-');
%legend('Donald 1 - The fixed image', 'Donald 2 - The moving image');
%axis ij; % This reverses the direction of the axis

mf = mean(fixedPoints);
mm = mean(movingPoints);

ex = mf-mm;

% Exam result
dist = sqrt(ex * ex')

mytform = fitgeotrans(movingPoints, fixedPoints,'NonreflectiveSimilarity');
donald2t = imwarp(donald2, mytform);

% exam result
resR = donald2t(300,300,1)
resG = donald2t(300,300,2)
resB = donald2t(300,300,3)


%subplot(1,2,1)
%imshow(donald1)
%title('donald 1')
%subplot(1,2,2)
%imshow(donald2)
%title('donald 2')

%figure;
%subplot(1,2,1)
%imshow(donald2t)
%title('donald2 transformed')
%subplot(1,2,2)
%imshow(donald1)
%title('donald1')


%% DICOM analysis
clc; clear; close all;

ct = dicomread('DICOM/1-131.dcm');
% imshow(ct, [-100, 200]);

LiverROI = imread('DICOM/LiverROI.png');
LiverVals = double(ct(LiverROI));

meanLiver = mean(LiverVals)
stdLiver = std(LiverVals)

SpleenROI = imread('DICOM/SpleenROI.png');
SpleenVals = double(ct(SpleenROI));

meanSpleen = mean(SpleenVals)
stdSpleen = std(SpleenVals)

BoneROI = imread('DICOM/BoneROI.png');
BoneVals = double(ct(BoneROI));

meanBone = mean(BoneVals)
stdBone = std(BoneVals)

% Exam result 1
T1 = (meanLiver + meanSpleen) / 2
T2 = (meanBone + meanSpleen) / 2

% Other segmentation question
T1 = 85; % Lower limit
T2 = 400; % Upper limit

binI = (ct > T1) & (ct < T2);
% figure
% imshow(binI);

se1 = strel('disk',5);
se2 = strel('disk',3);

e1 = imclose(binI, se1);

%figure;
%imshow(e1,[]);

e2 = imopen(e1, se2);

%figure;
%imshow(e2,[]);

%figure;
L8 = bwlabel(e2,8);
%imshow(label2rgb(L8))

% Start by calculating the area properties of the 8-connected components
stats8 = regionprops(L8, 'Area', 'Perimeter');

% It is more efficient to get all the measured areas out in one vector:
allArea = [stats8.Area];
allPerm = [stats8.Perimeter];

% exam result
idx = find([stats8.Area] > 1000 & [stats8.Area] < 4000);
BW2 = ismember(L8,idx);
figure;
imshow(BW2)

%% Median filter and linear mapping
clear,close all,clc

vb = imread('water/water_gray.png');
%imshow(vb);

fsize = 3;
medianim1 = medfilt2(vb,[fsize fsize]);

Itemp = double(medianim1);

% Stretch the image intensities
vmax_d = 230;
vmin_d = 12;

% Get vmax and vmin
vmin = min(Itemp(:));
vmax = max(Itemp(:));

Itemp = ((vmax_d - vmin_d) / (vmax - vmin)) * (Itemp - vmin) + vmin_d;
Io = uint8(Itemp);

%figure;
%imshow(Io);

% exam result
res = Io(20,20)


%% HSV2RBG and image filter with border replication
clear,close all,clc

I = imread('bird/bird.png');

%figure;
%imshow(I,[]);

HSV = rgb2hsv(I);
%figure;
%imshow(HSV)
Hcomp = HSV(:,:,1);
Scomp = HSV(:,:,2);
Vcomp = HSV(:,:,3);

%figure;
%imshow(Hcomp);

%figure;
%imshow(Scomp);

%figure;
%imshow(Vcomp);


h = fspecial('prewitt');
filt = imfilter(Scomp, h,'replicate');

% The answer
figure;
imshow(filt,[]);


%% BLOB analysis of climbing wall
clear,close all,clc

im = imread('Climbing/ClimbingWall.png');
imshow(im,[]);

Rcomp = im(:,:,1);
Gcomp = im(:,:,2);
Bcomp = im(:,:,3);

%imshow(Gcomp);

% RGB segmentation
Ibin =  Gcomp < 200 & Rcomp < 60 & Bcomp < 100;
%figure;
%imshow(Ibin);

se1 = strel('disk',3);
e1 = imclose(Ibin, se1);

%figure;
%imshow(e1,[]);

clearI = imclearborder(e1);
%figure;
%imshow(clearI,[]);

%figure;
L8 = bwlabel(clearI,8);
%imshow(label2rgb(L8))

% Exam answer 1 : how many blobs
res=max(max(L8))

% Start by calculating the area properties of the 8-connected components
stats8 = regionprops(L8, 'Area');

% It is more efficient to get all the measured areas out in one vector:
allArea = [stats8.Area];

P8 = regionprops(L8,'perimeter');
allPerimeter = [P8.Perimeter];

idx = find([stats8.Area] > 300 & [P8.Perimeter] < 500);
%BW2 = ismember(L8,idx);
%figure;
%imshow(BW2)

% exam answer 2:
res=size(idx,2)


%% Color thresholding
clear; clc; close all;
im = imread('BikeImage/bikes.png');
%figure;
%imshow(im);

Rcomp = im(:,:,1);
Gcomp = im(:,:,2);
Bcomp = im(:,:,3);

%imshow(Gcomp);

% RGB Treshold
segm =  Gcomp > 200 & Rcomp < 100 & Bcomp > 100;
%figure;
%imshow(segm);

se2 = strel('disk',8);
e1 = imclose(segm, se2);

% Exam answer:
figure;
imshow(e1);



%% PCA on images
clear; clc; close all;

N = 6;
H = 533;
W = 400;
M = H * W;

data = zeros(M,N);

img = imread('ImagePCA/orchid001.png');
tt = reshape(img, M, 1);
data(:,1) = tt;

img = imread('ImagePCA/orchid002.png');
tt = reshape(img, M, 1);
data(:,2) = tt;

img = imread('ImagePCA/orchid003.png');
tt = reshape(img, M, 1);
data(:,3) = tt;

img = imread('ImagePCA/orchid004.png');
tt = reshape(img, M, 1);
data(:,4) = tt;

img = imread('ImagePCA/orchid005.png');
tt = reshape(img, M, 1);
data(:,5) = tt;

img = imread('ImagePCA/orchid006.png');
tt = reshape(img, M, 1);
data(:,6) = tt;

meanI = mean(data, 2);
I = reshape(meanI, H, W);
%imshow(I,[]);

% Exam answer
imshow(I > 150);

[Vecs, Vals, Psi] = pc_evectors(data, 5);

v1 = Vecs(:,1);
v1img = reshape(v1, H, W);

% Exam result
resVal = v1img(10,10)

newflower = imread('ImagePCA/orchid007.png');
%imshow(newflower, [])
newMat = double(reshape(newflower, [], 1));

newMat = newMat - meanI;
Proj = Vecs(:, 1:1)' * newMat;

% Exam result
res=Proj(1)

%% PCA Analysis with pizza data
clear; close all; clc;
load PCAdata/pizza.txt;

% Data matrix
X = pizza';

[M,N] = size(X);
% M is number of features and N is number of observations

% Plot scatterplots of all variables:
%[~,ax]=plotmatrix(X');

% PCA computation:
data = X;

% subtracting mean for each dimension
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

% Exam answer
twofirst = (V(1) + V(2)) / sum(V) * 100

% First PC
pc1 = PC(:,1)

% Synthetic pizza
synth_piz = mn + 3 * pc1

% Exam answer 
res = synth_piz(1)


