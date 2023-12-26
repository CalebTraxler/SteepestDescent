f1 = @(x, y) x^2 + y - 11;
f2 = @(x, y) x + y^2 - 7;

F = @(x,y) [x^2 + y - 11; x + y^2 - 7];

tol = 1e-7;
max_iter = 1000;

n = 2;

p0 = -1;
q0 = 1;

df1dx = @(x, y) 2*x;
df1dy = @(x, y) 1;

df2dx = @(x, y) 1;
df2dy = @(x, y) 2*y;

J = @(x, y) [2*x, 1; 1, 2*y];

x0 = [p0; q0];

g = @(x) f1(x(1), x(2))^2 + f2(x(1), x(2))^2;

grad_g = @(x) 2*transpose(J(x(1), x(2))) * F(x(1), x(2));

k = 1;

while (k <= max_iter)
    d = -grad_g(x0);
    alpha = 1;
    s = 0.95;
    t = 0.45;

    while (g(x0 + alpha*d) > g(x0) - alpha*t*norm(grad_g(x0))^2)
        alpha = s*alpha;
    end

    alpha0 = alpha;

    x = x0 + alpha0*d;

    error = norm(grad_g(x0), inf);
    fprintf('Approximation: %f, %f at Iteration: %d, Error: %f\n', x(1), x(2), k, error);

    if (error < tol)
        fprintf('Solution: [%f; %f]\n', x(1), x(2));
        fprintf('\n');
        fprintf('Iterations: %d\n', k);
        break;
    end

    k = k + 1;

    x0 = x;
end

if (k > max_iter)
    fprintf('Method did not converge within %d iterations.\n', max_iter);
end