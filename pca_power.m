function pca_power(training_file, test_file, M, iterations)
training_file = "../dataset/satellite_training.txt";
test_file = "../dataset/satellite_test.txt";
M = 2;
iterations = 20;
f = load(training_file, "-ascii");
D = size(f,2)-1;

x = f(:,1:D);
xd = x; % N,D
N = size(xd, 1);
U = zeros(M, D);

for dd = 1:M
    xbar = mean(xd)'; % D,1
    Sd = zeros(D, D);
    for i = 1:N
        % (Dx1 - Dx1) * (Dx1 - Dx1)' => Dx1 * 1xD = DxD
        Sd += (xd(i,:)' - xbar) * (xd(i,:)' - xbar)';
    end
    Sd = Sd ./ N;
    b = rand(D, 1); % Dx1
    for j = 1:iterations
        b = Sd * b; %
        b = b ./ norm(b, 2);  
    end
    ud = b;
    for i = 1:N
        xd(i,:) = xd(i,:) - (ud' * xd(i,:)' * ud)';
    end
    U(dd,:) = ud';        
    printf("Eigenvector %d\n", dd);
    for i = 1:D
        printf("%3d: %.4f\n", i, ud(i));
    end
    printf("\n");
    %ud'
    %if dd == 1
    %    [V, lam] = eig(Sd);
    %end
    %V(:,D-dd+1)'
    %lam
    %max((Sd - cov(x, 1)) ./ cov(x, 1))
end

tst = load(test_file, "-ascii");

x_tst = tst(:,1:D);
tst_size = size(tst,1);

for i = 1:tst_size
    printf("Test object %d\n", i-1);
    F = U * x_tst(i,:)';
    for j = 1:M
        printf("%d: %.4f\n", j, F(j));
    end
    printf("\n");
end
    
end
