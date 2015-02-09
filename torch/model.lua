
function model(X_train, y_train, X_test)
    proba = torch.rand(X_test:size(1), X_test:size(2))
    targets = torch.gt(proba, 0.5)
    return targets, proba
end
