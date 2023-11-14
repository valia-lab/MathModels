using CSV
using DataFrames
using JuMP
using GLPK

data = CSV.read("data.csv", DataFrame)
println(data)
N = nrow(data)
active_variables::Int64 = 0 
current_year = 2022
current_month = 11
vars = Dict()

for record in eachrow(data)
    Date = record.Date
    if ismissing(Date) #|| isMissing(record.Price) || isMissing(record.Country))
        continue          #  empty 
    else
        d, m, y = split(Date, "/")
        if parse(Int64,y) > current_year
            continue
        elseif parse(Int64, m) > current_month
            continue
        else 
            global active_variables += 1
            vars[record.Country] = record.Price, record.Date
        end
    end
end  

prices = Array{Int64}(undef, active_variables)
i = 1
for k in keys(vars)
    prices[i] = vars[k][1]
    global i += 1
end

M = maximum(prices)
countries = collect(keys(vars) )

# Define the model
m = Model(); # m stands for model

# Set the optimizer
set_optimizer(m, GLPK.Optimizer); # calling the GLPK solver

# Define the variables
@variable(m, x[1:  active_variables], Bin);
@variable(m, z[1:  active_variables], Bin);     # z = 1 iff i-th country is in the 3 cheapest countries
@variable(m, q[1:  3], Bin);          # q is the number of countries in the cost function : q = 1, 2, 3
@variable(m, r[1:  3]);          # r is the price ratio if q countries are in the cost function
               

# Define the constraints 
@constraint(m, sum(z)<= 3);
@constraint(m, sum(x)>= 2);
@constraint(m, [i = 1:active_variables], z[i] <= x[i])
@constraint(m, [i = 1:active_variables], sum(z) <= sum(x) )
@constraint(m, [i=1:active_variables, j=1:active_variables], (z[i]-1)*prices[i] <= (1-z[j])*prices[j])

@constraint(m, [i = 1:active_variables], sum(q) == 1)   #only one possible event
@constraint(m, [i=1:active_variables], sum(z) == sum(i * q[i] for i in 1:3))
@constraint(m, [i = 1:3], r[i] <= M*q[i])
@constraint(m, [i = 1:3], r[i] <= 1/i*sum(prices[j]*z[j] for j in 1:active_variables)) 


# Define the objective function
@objective(m, Max, sum(r));

# Run the solver
optimize!(m);

# Output
for i in 1:active_variables
    if value.(x[i]) == 1
        println("x[", i, "] = ", value.(x[i]), " : ", countries[i], " was not delayed.")
    else 
        println("x[", i, "] = ", value.(x[i]), " : ", countries[i], " was delayed")
    end
end

for i in 1:3
    if i==3 && value.(q[1]) == 1
        println(value.(q[i]), " : country are in the cost function.")
    elseif && i>1 value.(q[i])==1
        println(value.(q[i]), " : countries are in the cost function.")
    end
end

println("The final price was ", objective_value(m)) # optimal value z
#println("x = ", value.(x), "\n","y = ",value.(y)) # optimal solution x & y

for i in 2:result_count(m)
    @assert has_values(model; result = i)
    println("Solution $(i) = ", value.(x; result = i))
    obj = objective_value(model; result = i)
    println("Objective $(i) = ", obj)
    if isapprox(obj, optimal_objective; atol = 1e-8)
        print("Solution $(i) is also optimal!")
    end
end
