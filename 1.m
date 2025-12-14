

clear; clc;


%----------Gradient Method----------
function [x,mse] = gradient_method(A,b,u_exact,maxit,tol)
n = length(b);
x = zeros(n,1);
mse = zeros(maxit,1);

r = b - A*x;
r0 = r'*r;

for k = 1:maxit
    Ar = A*r;
    alpha = (r'*r)/(r'*Ar);
    x = x + alpha*r;

    r = r - alpha*Ar;
    mse(k) = mean((x - u_exact).^2);

    if (r'*r)/r0 < tol^2
        mse = mse(1:k);
        break
    end
end
end




%----------Conjugate Direction Method----------
function [x,mse] = conjugate_direction(A,b,u_exact)
n = length(b);
x = zeros(n,1);

% number of directions = n
V = eye(n);
P = zeros(n,n);

mse = zeros(n,1);
r = b - A*x;

for k = 1:n
    v = V(:,k);
    p = v;

    for j = 1:k-1
        pj = P(:,j);
        p = p - (pj'*A*v)/(pj'*A*pj) * pj;
    end

    P(:,k) = p;

    alpha = (r'*p)/(p'*A*p);
    x = x + alpha*p;
    r = b - A*x;
    mse(k) = mean((x - u_exact).^2);

    if norm(p) < 1e-14
      break
    end
end
end






%----------Conjugate Gradient Method----------
function [x,mse] = conjugate_gradient(A,b,u_exact,maxit,tol)
n = length(b);
x = zeros(n,1);
r = b - A*x;
p = r;
mse = zeros(maxit,1);

for k = 1:maxit
    alpha = (r'*r)/(p'*A*p);
    x = x + alpha*p;
    r_new = r - alpha*A*p;
    mse(k) = mean((x-u_exact).^2);
    if norm(r_new) < tol
        mse = mse(1:k);
        break
    end
    beta = (r_new'*r_new)/(r'*r);
    p = r_new + beta*p;
    r = r_new;
end
end




%----------Preconditioned Conjugate Gradient Method----------
function [x,mse] = pcg_jacobi(A,b,u_exact,maxit,tol)
n = length(b);
x = zeros(n,1);
M = diag(diag(A));

r = b - A*x;
z = M\r;
p = z;
mse = zeros(maxit,1);

for k = 1:maxit
    alpha = (r'*z)/(p'*A*p);
    x = x + alpha*p;
    r_new = r - alpha*A*p;
    mse(k) = mean((x-u_exact).^2);
    if norm(r_new) < tol
        mse = mse(1:k);
        break
    end
    z_new = M\r_new;
    beta = (r_new'*z_new)/(r'*z);
    p = z_new + beta*p;
    r = r_new;
    z = z_new;
end
end



%----------Main Function----------
hs = [0.1, 0.01, 0.005];

for hh = 1:length(hs)
    h = hs(hh);
    n = 1/h - 1;
    x = (1:n)'*h;

    % Exact solution
    u_exact = 1 + (x.^2).*(1-x).^2;

    % Build A matrix
    A = zeros(n,n);
    for i = 1:n
        A(i,i) = -2;
        if i > 1
            A(i,i-1) = 1;
        end
        if i < n
            A(i,i+1) = 1;
        end
    end
    A = A / h^2;

    % RHS
    f = 12*(x.^2) - 12*x + 2;
    b = f;
    b(1) = b(1) - 1/h^2;
    b(end) = b(end) - 1/h^2;

    maxit = 1000;
    tol = 1e-10;

    [x1,mse1] = gradient_method(A,b,u_exact,maxit,tol);
    [x2,mse2] = conjugate_direction(A,b,u_exact);
    [x3,mse3] = conjugate_gradient(A,b,u_exact,maxit,tol);
    [x4,mse4] = pcg_jacobi(A,b,u_exact,maxit,tol);

    figure;
    semilogy(mse1,'LineWidth',1.5,'MarkerSize',4); hold on;
    semilogy(mse2,'--', 'LineWidth',1.5,'MarkerSize',4);
    semilogy(mse3,'LineWidth',1.5,'MarkerSize',4);
    semilogy(mse4, "--", 'LineWidth',1.5,'MarkerSize',4);

    grid on;
    legend('Gradient','Conjugate Dir','CG','PCG','Location','southwest');
    xlabel('Iteration');
    ylabel('MSE (log scale)');
    ylim([1e-3 1e+1])
    title(['h = ',num2str(h)]);


end

