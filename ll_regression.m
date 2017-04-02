function ll_regression(training_file, degree, test_file)
f = load(training_file, "-ascii");
D = size(f,2)-1;

x = f(:,1:D);
t = f(:,D+1);
t = (t == 1);
classification_accuracy = 0;

if (degree == 1)
	phi = [ones(size(x,1),1) x];
	w = zeros(D+1, 1);
%	size (w)
%	size (phi)
	err = 10;
	cee = 10;
	while ((err > 0.001) && (cee > 0.001))
		y = sigmoid(phi * w);
		Rnn = y .* (1-y);
		R = diag(Rnn);
		wn = w - inv(phi' * R * phi) * phi' * (y - t);
		err = sum(abs(wn - w));
		w = wn;
		cee = -mean(t .* log(y) + (1 - t) .* log(1-y));
%		printf("%f  %f\n", err, cee);		
	end

	for n = 0:D
		printf("w%d=%.4f\n", n, w(n+1));
	end		

	tst = load(test_file, "-ascii");

	x_tst = tst(:,1:D);
	t_tst = tst(:,D+1);
	t_tst = (t_tst == 1);
	tst_size = size(tst,1)
	phi_tst  = [ones(size(x_tst,1),1) x_tst];
	y_tst = sigmoid(phi_tst * w);
	classification_accuracy = 0;	
%tst size is required

	for n = 1:tst_size
		prob = y_tst(n);
		if (prob > 0.5)
			pred = 1;
		else
			pred = 0;
			prob = 1 - prob;		 
		end		

		if (pred == t_tst(n))
			accu = 1;
		else
			accu = 0;
		end

		if (prob == 0.5)
			pred = rand(1, 1);
			if (pred > 0.5)
				pred = 1;
			else
				pred = 0;
			end	
			accu = 0.5;
		end
		classification_accuracy += accu / tst_size; 
		
		printf("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n", 
	       n-1, pred, prob, t_tst(n), accu);

	end	
	printf("classification accuracy=%6.4f\n", classification_accuracy);

else 

	x2 = x .^ 2; 
	x_x2 = reshape([x'(:) x2'(:)]', 2 * size(x', 1), [])';

	phi = [ones(size(x,1),1) x_x2];
	w = zeros(2 * D+1, 1);
%	size (w)
%	size (phi)
	err = 10;
	cee = 10;

	while ((err > 0.001) && (cee > 0.001))
		y = sigmoid(phi * w);
		Rnn = y .* (1-y);
		R = diag(Rnn);
		wn = w - inv(phi' * R * phi) * phi' * (y - t);
		err = sum(abs(wn - w));
		w = wn;
		cee = -mean(t .* log(y) + (1 - t) .* log(1-y));
		printf("%f  %f\n", err, cee);		
	end

	for n = 0:(2 * D)
		printf("w%d=%.4f\n", n, w(n+1));
	end		

	tst = load(test_file, "-ascii");

	x_tst = tst(:,1:D);
	t_tst = tst(:,D+1);
	t_tst = (t_tst == 1);
	tst_size = size(tst,1)

	x2_tst = x_tst .^ 2; 
	x_x2_tst = reshape([x_tst'(:) x2_tst'(:)]', 2 * size(x_tst', 1), [])';
	phi_tst = [ones(size(x_tst,1),1) x_x2_tst];
	y_tst = sigmoid(phi_tst * w);
	classification_accuracy = 0;	

	for n = 1:tst_size
		prob = y_tst(n);
		if (prob > 0.5)
			pred = 1;
		else
			pred = 0;
			prob = 1 - prob;		 
		end		

		if (pred == t_tst(n))
			accu = 1;
		else
			accu = 0;
		end

		if (prob == 0.5)
			pred = rand(1, 1);
			if (pred > 0.5)
				pred = 1;
			else
				pred = 0;
			end	
			accu = 0.5;
		end
		classification_accuracy += accu / tst_size; 
		
		printf("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n", 
	       n-1, pred, prob, t_tst(n), accu);

	end	
	printf("classification accuracy=%6.4f\n", classification_accuracy);

end

end

%Sigmoid function
function y = sigmoid(x)
	y = 1 ./ (1 + e.^-x);
end