function model(X_train, y_train, X_test)
    proba = torch.rand(X_test:size(1), 2)
    targets = argmax_2D(proba) - 1
    return targets, proba
end
