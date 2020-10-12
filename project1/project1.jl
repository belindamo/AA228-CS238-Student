using LightGraphs
using Printf
using Random
using DelimitedFiles
using SpecialFunctions
using LinearAlgebra

"""
  write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function write_gph(dag::DiGraph, idx2names, filename)
  open(filename, "w") do io
    for edge in edges(dag)
        @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
    end
  end
end

# ---------- Textbook Code ----------
function sub2ind(siz, x)
  k = vcat(1, cumprod(siz[1:end-1]))
  return dot(k, x .- 1) + 1
end

function statistics(vars, G, D::Matrix{Int})
  n = size(D, 1)
  # print(n)
  r = [length(vars[i]) for i in 1:n]
  q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
  M = [zeros(q[i], r[i]) for i in 1:n]
  for o in eachcol(D)
    for i in 1:n
      k = o[i]
      parents = inneighbors(G,i)
      j = 1
      if !isempty(parents)
        j = sub2ind(r[parents], o[parents])
      end
      M[i][j,k] += 1.0
    end
  end
  return M
end

function prior(vars, G)
  n = length(vars)
  r = [length(vars[i]) for i in 1:n]
  q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
  return [ones(q[i], r[i]) for i in 1:n]
end

function bayesian_score_component(M, α)
  p = sum(loggamma.(α + M)) # loggamma is imported by specialfunctions
  p -= sum(loggamma.(α))
  p += sum(loggamma.(sum(α,dims=2)))
  p -= sum(loggamma.(sum(α,dims=2) + sum(M,dims=2)))
  return p
end

function bayesian_score(vars, G, D)
  n = length(vars)
  M = statistics(vars, G, D)
  α = prior(vars, G)
  return sum(bayesian_score_component(M[i], α[i]) for i in 1:n)
end

struct K2Search
  ordering::Vector{Int} # variable ordering
end


# vars: ordered list of variables
# D: dataset
# m_ijk = number of times in a dataset the variable x_i takes on its  
# k-th value when the x_i's parents take on their j-th value
function fit(method::K2Search, vars, D)
  G = SimpleDiGraph(length(vars))
  # show(stdout, "text/plain", G) 
  for (k,i) in enumerate(method.ordering[2:end]) # i = child node index, k = child value index
    y = bayesian_score(vars, G, D)
    while true
      y_best, j_best = -Inf, 0
      for j in method.ordering[1:k] # j = parent value index
        if !has_edge(G, j, i)
          add_edge!(G, j, i)
          y′ = bayesian_score(vars, G, D)
          if y′ > y_best
            y_best, j_best = y′, j
            # print(y′)
          end
          rem_edge!(G, j, i)
        end
      end
      if y_best > y
        y = y_best
        add_edge!(G, j_best, i)
      else
        break
      end
    end
  end
  return G
end

# ---------- Compute ----------
# struct Var 
#   set::Set{Int}
#   total_values::Int
# end

function compute(infile, outfile)
  filedata = readlines(infile)
  names = split(filedata[1], ",")
  num_vars = length(names)
  idx2names = Dict{Int, String}()
  # vars = zeros(Int64, num_vars)
  vars = Array{Set}(undef, num_vars)
  for i = 1:num_vars
    vars[i] = Set{Int}()
    idx2names[i] = names[i]
  end
  # D = filedata[2:end]
  # D = 
  # D = readdlm(infile, "\,", Int, "\n")
  # D = zeros(length(filedata)-1, num_vars)
  
  D = Matrix{Int64}(undef, num_vars, length(filedata)-1)
  # for (row_i, row_arr) in enumerate(filedata[2:end])
  #   row = split(row_arr, ",")
  #   # x = [parse(Int, x) for x in row]
  #   for (col_i, val) in enumerate(row)
  #     x = parse(Int, val)
  #     D[row_i, col_i] = x
  #     push!(vars[col_i], x)
  #   end
  # end  
  for (col_i, col_arr) in enumerate(filedata[2:end])
    col = split(col_arr, ",")
    for (row_i, val) in enumerate(col)
      x = parse(Int, val)
      D[row_i, col_i] = x
      push!(vars[row_i], x)
    end
  end
  # D = copy(transpose(D))

  # show(stdout, "text/plain", vars) 
  # show(stdout, "text/plain", D) 
  ordering = shuffle(collect(1:1:num_vars))
  # show(stdout, "text/plain", ordering) 
  k2search = K2Search(ordering)
  G = fit(k2search, vars, D)
  # show(stdout, "text/plain", G) 

  write_gph(G, idx2names, outfile)
  final_score = bayesian_score(vars, G, D)
  @printf(stdout, "Final bayesian score: %f\n", final_score)
end


# ---------- Execute ----------
if length(ARGS) != 2
  error("usage: julia project1.jl <infile>.csv <outfile>.gph")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

@time compute(inputfilename, outputfilename)
