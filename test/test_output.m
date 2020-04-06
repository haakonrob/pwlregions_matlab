load('net.mat')




N = 500;
X = 5*randn(2, N);
box = Polyhedron('lb', [min(X(1,:)), min(X(2,:))], 'ub', [max(X(1,:)), max(X(2,:))]);
regs = pwl_struct(net, 'input_space', box);

Y_true = eval_struct(net,X);
Y = zeros(size(X));

for i = 1:N
    % MPT doesn't support evaluating multiple points at once
    [v, feasible] = regs.feval(X(:,i), 'f');
    if any(feasible)
        Y(:,i) = v(:,feasible);    
    end
end

err = norm((Y-Y_true).^2, 2)

val(:,feasible);

%%
% First outputs
figure(1)
subplot(2,1,1);
scatter3(X(1,:), X(2,:), Y_true(1,:), 'r.')
axis equal

subplot(2,1,2);
regs.fplot('f1');
axis equal

% Second outputs
figure(2)
subplot(2,1,1);
scatter3(X(1,:), X(2,:), Y_true(2,:), 'r.')
axis equal

subplot(2,1,2);
regs.fplot('f2');
axis equal