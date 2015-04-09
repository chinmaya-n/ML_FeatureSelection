 x <- c(1, 5, 100, 1000, 3000, 4000, 5000, 10000, 14000, 20000)
 y <- c(2,4,6,8,7,12,14,16,18,20)

 smoothingSpline = smooth.spline(x, y, spar=0.35)
plot(x,y)
lines(smoothingSpline)