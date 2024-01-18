using PlotlyJS,  DataFrames
n = 300
d = 0.05
X = LinRange(-6,6,n)
Y = LinRange(-6,6,n)
Z = zeros(n,n)
eps = 0.5

for i in 1:n
    for j in 1:n
        x = X[i]
        y = Y[j]
        if x <= 3 &&  y >= -1 && (- 2*x - 3*y) <= 6 && (-x + 2*y) <= 6
            Z[i, j] = 2*x - y
        end
    end
end


    
    
    
    
z_data = Matrix{Float64}(Z)'

layout = Layout(
    title="Задача 4",
    autosize=false,
    scene_camera_eye=attr(x=1.87, y=0.88, z=-0.64),
    width=1000, height=1000,
    margin=attr(l=65, r=50, b=65, t=90)
)
p = plot(surface(
    z=Z,
    x = X,
    y = Y,
    contours_z=attr(
        show=true,
        usecolormap=true,
        highlightcolor="limegreen",
        project_z=true
    )
), layout)

open("./example.html", "w") do io
    PlotlyBase.to_html(io, p.plot)
end
