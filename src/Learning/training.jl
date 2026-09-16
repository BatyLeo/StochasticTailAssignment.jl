"""
$TYPEDSIGNATURES

Build a default feedforward neural network mapping `nb_features` input features to a
single scalar edge weight per arc, with hidden layers of dimensions `hidden_dims`.
"""
function default_model(nb_features; hidden_dims=[10, 10])
    layers = []
    input_dim = nb_features
    for h in hidden_dims
        push!(layers, Dense(input_dim => h, relu))
        input_dim = h
    end
    push!(layers, Dense(input_dim => 1))
    push!(layers, vec)
    return Chain(layers...)
end

"""
$TYPEDSIGNATURES

Train `model` by imitation of the expert solutions in `data_train`, using the Fenchel-Young
loss `fyl_loss` (see [`build_loss`](@ref)) and evaluating with `maximizer` on `data_train`
and `data_val` after each epoch.

At each epoch (including epoch 0, before any update), the validation full cost gap
(`metrics_val[epoch].full_cost_gap`) is compared to the best value seen so far, and a
`deepcopy` of `model` is checkpointed whenever it improves.

Returns a `NamedTuple` with fields `loss_history`, `metrics_train`, `metrics_val`,
`best_model` (a `deepcopy` of `model` at the epoch with the lowest validation full cost
gap) and `best_epoch` (that epoch number, `0` meaning before any training step).
"""
function train_model!(
    model, fyl_loss, maximizer, data_train, data_val; nb_epochs=10, relaxation=false
)
    opt = Adam()
    opt_state = Flux.setup(opt, model)
    loss_history = Float64[]
    metrics_train = NamedTuple[]
    metrics_val = NamedTuple[]

    push!(
        loss_history,
        mean(fyl_loss(model(e.x), e.ȳ; e.instance, relaxation) for e in data_train),
    )
    push!(metrics_train, evaluate_metrics(data_train, model, maximizer))
    m_val = evaluate_metrics(data_val, model, maximizer)
    push!(metrics_val, m_val)

    best_epoch = 0
    best_model = deepcopy(model)
    best_val_gap = m_val.full_cost_gap

    for epoch in 1:nb_epochs
        loss = 0.0
        for e in data_train
            (; instance, x, ȳ) = e
            l = 0.0
            grads = gradient(model) do m
                θ = m(x)
                l = fyl_loss(θ, ȳ; instance, relaxation)
                return l
            end
            loss += l
            Flux.update!(opt_state, model, grads[1])
        end
        push!(loss_history, loss / length(data_train))
        push!(metrics_train, evaluate_metrics(data_train, model, maximizer))
        m_val = evaluate_metrics(data_val, model, maximizer)
        push!(metrics_val, m_val)

        if m_val.full_cost_gap < best_val_gap
            best_val_gap = m_val.full_cost_gap
            best_epoch = epoch
            best_model = deepcopy(model)
        end
    end

    return (; loss_history, metrics_train, metrics_val, best_model, best_epoch)
end
