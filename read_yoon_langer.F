      program read_yoon_langer

! A short program to read in Yoon & Langer's (2005) paper data files,
!  written in Fortran from Sung-Chul's IO routines.

      implicit none

      integer i, j, nth, nr, nmd
      real*8  G, pi, rmax, mtot, rhomax, tw, jtot, egrav, erot, pi3, & 
            frhocrit, w0, ws, bfit, dV, totaleg, msun, massfac, &
            timefac, lengthfac, temp, dM, totalM
      real*8  mu (149), x (149), dcosth (149), dx (149)
      real*8  rho (149, 149), w (149, 149), psi (149, 149), phi (149, 149)

      pi = 4. * atan (1.)
      G = 6.674e-8 ! CGS units
      msun = 1.987e33 ! CGS

! The output format in each file is as the following:

      open(17,file="plot01", action='read', status='old')
      read(17,220) nth, nr, nmd
      read(17,221) rmax,mtot,rhomax,tw,jtot,egrav,erot,pi3, &
              temp,frhocrit,w0,ws,bfit
      do i = 1,nth
      do j = 1, nr
        read(17,222) rho(i,j), w(i,j),psi(i,j), phi(i,j)
      enddo
      enddo
 220  format(1x,i3, 1x, i3, 1x, i1)
 221  format(13(1x,d20.9))
 222  format(4(1x,d20.9))
      close(17)

!    Coordinates -----------------------------------------------
      do i=1,nth
        mu(i) = (i-1.d0)/(nth - 1.0d0) ! cos(th)
      enddo
      do j=1,nr
        x(j) = 16.*(j-1.d0)/(nr-1.0)/15.0
      enddo

! Compute globally-integrated quantities

      do i = 1, nth
        if (i .ne. 1) then
          dcosth (i) = mu (i) - mu (i - 1)
        else
          dcosth (i) = mu (i + 1) - mu (i)
        endif
      end do ! loop over nth

      do j = 1, nr
        if (j .ne. 1) then
          dx (j) = x (j) - x (j - 1)
        else
          dx (j) = x (j + 1) - x (j)
        end if
      end do

      totalM  = 0.
      totaleg = 0.
      
      do i = 1, nth
        do j = 1, nr
   
          dV = 4. * pi * (x (j) )**2. * dcosth (i) * dx (j)
          dM = rho (i, j) * dV
          totalM  = totalM + dM
          totaleg = totaleg + dM * phi (i, j) 

        end do
      end do

      massfac   = rhomax * rmax**3.
      lengthfac = rmax
      timefac   = (G * rhomax)**(-1./2.)

      totalM    = totalM    * massfac
      totaleg   = totaleg   * massfac * lengthfac**2. / timefac**2.

      print *, "Total computed mass = ", totalM
      print *, "Stored mass = ", mtot
      print *, "Total computed Eg = ", totalEg
      print *, "Stored Eg = ", egrav

      end
