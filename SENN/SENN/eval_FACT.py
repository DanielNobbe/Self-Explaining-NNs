


def main():

	# Load Model

	checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'), map_location=lambda storage, loc: storage)
	checkpoint.keys()
	model = checkpoint['model']
	trainer =  VanillaClassTrainer(model, args)

	# Do validation
    trainer.validate(test_loader, fold = 'test')


    print(test_tds)

    distances = defaultdict(list)

    print(distances)

    scale = 0.5

    print(distances)

    for i in tqdm(range(1000)):

	    x = Variable(test_tds[i][0].view(1,1,28,28), volatile = True)

	    true_class = test_tds[i][1][0].item()

	    pred = model(x)

	    theta = model.thetas.data.cpu().numpy().squeeze()

	    klass = pred.data.max(1)[1]
	    deps = theta[:,klass].squeeze()

	    # print("prediction", klass)
	    # print("dependencies", deps)

	    # Add noise to sample and repeat
	    noise = Variable(scale*torch.randn(x.size()), volatile = True)

	    pred = model(noise)

	    theta = model.thetas.data.cpu().numpy().squeeze()

	    klass_noise = pred.data.max(1)[1]
	    deps_noise = theta[:,klass].squeeze()

	    dist = np.linalg.norm(deps - deps_noise)

	    distances[true_class].append(dist)

    print(distances)

if __name__ == '__main__':
    main()