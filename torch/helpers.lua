local helpers = {}

function helpers.get_data_and_column_names(data_filename)
    X, column_names = csv2tensor.load(data_filename, {exclude='TARGET'})
    y, dummy = csv2tensor.load(data_filename, {include='TARGET'})
    return({X=X, y=y, column_names=column_names})
end

function helpers.data_overview(X, column_names, nb_rows, nb_cols)
    
    if nb_rows == nil then
        nb_rows = X:size(1)
    end

    if nb_cols == nil then
        nb_cols = X:size(2)
    end

    html = "<table>"
    html = html .. "<tr>"
    for i = 1, nb_cols do
        html = html .. "<td><strong>" .. column_names[i] .. "</strong></td>"
    end
    html = html .. "</tr>"

    for i = 1, nb_rows do
        html = html .. "<tr>"
        for j = 1, nb_cols do
            html = html .. "<td>" .. X[{i, j}] .. "</td>"
        end
        html = html .. "</tr>"
    end

    return(html)
end


function helpers.split_data(X, y, ratio)
    nb_elements_first = math.floor(ratio * X:size(1))
    remaining = X:size(1) - nb_elements_first
    X_first = X:narrow(1, 1, nb_elements_first)
    X_last = X:narrow(1, nb_elements_first + 1, remaining)

    y_first = y:narrow(1, 1, nb_elements_first)
    y_last = y:narrow(1, nb_elements_first + 1, remaining)
    return X_first, y_first, X_last, y_last
end


return helpers
