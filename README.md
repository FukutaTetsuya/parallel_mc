# parallel_mc
parallel update monte carlo simulation of 2D Manna model in continuous media<br>
periodic boundary condition, random initial condition,<br>
unit length a = particle diameter = 1.0<br>
<br>
a particle kicked randomly if it is "active", i.e. it has overlap with other particle(s)<br>
kick distance = rand[0.0,0.5*diametar], kick direction = rand[0.0, 2.0*pi)<br>
and acceptance ratio of kicks is 100 percent<br>
