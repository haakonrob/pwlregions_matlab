%% Init 
% rng(314);

%% Setup maxout(Wx+b) with n neurons
n = 4;
W = randn(n,2);
b = randn(n,1);

%% Utility functions to apply max to 3 hyperplanes

f = @(xx_, yy_, W_, b_) reshape(max(W_*[xx_(:),yy_(:)]'+b_), size(xx_));

%%  Plot the function
width = 100;
x = linspace(-width,width,100);
[xx,yy] = meshgrid(x,x);


% p = linsolve(W,-b)

figure(1)
surf(xx,yy,f(xx, yy, W, b));hold on;
% scatter3(p(1,:),p(2,:),W(1,:)*p + b(1), 'r')
hold off



%% Non trivial fixed example

% This example shows that maxout can lead to pretty complex regions. Here
% we make a kind of skew upside down pyramid with the first 4 rows, and
% then we add a flat hyperplane to "cut off" the corner of the pyramid.
% Pretty cool! The question is, how on earth do you find these regions?

W2 = [ 
   -1.0378    0.1579
    1.4795    0.5493
   -0.0673   -0.5578
    1.6434    0.0488
    0         0        % This last row is just a flat xy hyperplane that we lift up to "cut off" the corner
];

b2 = [
    0.1868
   -1.4244
   -0.3868
    0.8913
    10                 % This controls the height of the flat hyperplane
];

figure(2)
surf(xx,yy,f(xx, yy, W2, b2));hold on;
% scatter3(p(1,:),p(2,:),W(1,:)*p + b(1), 'r')
hold off