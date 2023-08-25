program main

    implicit none

    character(100), parameter ::    mesh        = 'circular.su2'    ! mesh file
    character(100), parameter ::    data        = 'data_circular/'  ! data folder
    integer, parameter ::           precision   = 8                 ! real number precision
    integer, parameter ::           interval    = 100               ! write interval
    real(precision), parameter ::   g           = 9.81              ! gravitational acceleration
    real(precision), parameter ::   dry         = 1e-6              ! dry depth
    real(precision), parameter ::   n           = 0.                ! Manning coefficient
    real(precision), parameter ::   tt          = 10.               ! total time
    real(precision), parameter ::   dt          = 1e-3              ! time step

    ! check initial conditions

    ! program start

    integer :: n_cells, n_faces, n_nodes, n_inflow_faces, n_outflow_faces, n_solid_faces, n_boundary_faces, n_interior_faces
    integer, dimension(:), allocatable :: inflow_faces, outflow_faces, solid_faces, boundary_faces, interior_faces, n_node_cells
    integer, dimension(:, :), allocatable :: cells, faces, cell_faces, face_cells, face_faces, cell_cells
    integer, dimension(:, :), allocatable :: cell_face_faces, cell_nodes, node_cells
    real(precision), dimension(:, :), allocatable :: nodes

    real(precision), dimension(:), allocatable :: Sf, Vc
    real(precision), dimension(:, :), allocatable :: nf, cf, cc, sc, rfc, rnc, dn, wn

    real(precision), dimension(:), allocatable :: h, hn, qx, qxn, qy, qyn, z, zn, b
    real(precision), dimension(:, :), allocatable :: hL, qxL, qyL, zL, bL, F, Q, Qs, Ks

    integer :: it
    real(precision) :: t

    ! create mesh

    call set_mesh()

    ! allocate variables

    allocate(h(n_cells))
    allocate(hn(n_nodes))
    allocate(hL(n_cells, 3))

    allocate(qx(n_cells))
    allocate(qxn(n_nodes))
    allocate(qxL(n_cells, 3))

    allocate(qy(n_cells))
    allocate(qyn(n_nodes))
    allocate(qyL(n_cells, 3))

    allocate(z(n_cells))
    allocate(zn(n_nodes))
    allocate(zL(n_cells, 3))

    allocate(b(n_cells))
    allocate(bL(n_cells, 3))

    allocate(F(n_faces, 3))
    allocate(Q(n_cells, 3))
    allocate(Qs(n_cells, 3))
    allocate(Ks(n_cells, 3))

    ! initial conditions

    call set_initial()

    ! time loop

    t = 0.
    it = 0
    do while (t < tt)
        
        call set_friction()

        Q(:, 1) = h(:)
        Q(:, 2) = qx(:)
        Q(:, 3) = qy(:)
        z(:) = b(:) + h(:)
    
        call set_reconstruction()
        call set_convection()
        call set_slope()
    
        Qs(:, 1) = h(:)  ! h*
        Qs(:, 2) = qx(:)  ! qx*
        Qs(:, 3) = qy(:)  ! qy*
        z(:) = b(:) + h(:)  ! z*
    
        call set_reconstruction()
        call set_convection()
        call set_slope()
    
        Q = .5*((Q + Qs) + Ks)
    
        h(:) = Q(:, 1)
        qx(:) = Q(:, 2)
        qy(:) = Q(:, 3)
        z(:) = b(:) + h(:)

        call set_data()
        
        t = t + dt
        it = it + 1
        print *, t, maxval(h), minval(h)

    end do

contains
    
    subroutine set_mesh()

        character(100) :: line, boundary
        integer :: io, i, j, k, n1, n2, n3, na, nb, nc, fi, fj, c1, c2, cj
        integer, dimension(:), allocatable :: l1, l2, l3
        real(precision) :: nx, ny, n1x, n1y, n2x, n2y, n3x, n3y, ncx, ncy, ccx, ccy, cfx, cfy

        open(10, file=trim(mesh), action='read')

        ! cells

        read(10, *)
        read(10, *) line, n_cells
        allocate(cells(n_cells, 3))
        do i = 1, n_cells
            read(10, *) line, n1, n2, n3
            cells(i, :) = [n1 + 1, n2 + 1, n3 + 1]
        end do

        ! nodes

        read(10, *) line, n_nodes
        allocate(nodes(n_nodes, 2))
        do i = 1, n_nodes
            read(10, *) nx, ny
            nodes(i, :) = [nx, ny]
        end do

        ! faces

        l1 = [integer ::]
        l2 = [integer ::]
        n_inflow_faces = 0
        do
            read(10, *, iostat=io) line, boundary
            if (boundary == 'INFLOW') then
                read(10, *) line, n_inflow_faces
                do i = 1, n_inflow_faces
                    read(10, *) line, n1, n2
                    l1 = [l1, n1 + 1]
                    l2 = [l2, n2 + 1]
                end do
                exit
            end if
            if (io /= 0) exit
        end do
        rewind(10)

        n_outflow_faces = 0
        do
            read(10, *, iostat=io) line, boundary
            if (boundary == 'OUTFLOW') then
                read(10, *) line, n_outflow_faces
                do i = 1, n_outflow_faces
                    read(10, *) line, n1, n2
                    l1 = [l1, n1 + 1]
                    l2 = [l2, n2 + 1]
                end do
                exit
            end if
            if (io /= 0) exit
        end do
        rewind(10)

        n_solid_faces = 0
        do
            read(10, *, iostat=io) line, boundary
            if (boundary == 'SOLID') then
                read(10, *) line, n_solid_faces
                do i = 1, n_solid_faces
                    read(10, *) line, n1, n2
                    l1 = [l1, n1 + 1]
                    l2 = [l2, n2 + 1]
                end do
                exit
            end if
            if (io /= 0) exit
        end do
        rewind(10)

        do i = 1, n_cells
            n1 = cells(i, 1)
            n2 = cells(i, 2)
            n3 = cells(i, 3)
    
            call set_face(l1, l2, n1, n2)
            call set_face(l1, l2, n2, n3)
            call set_face(l1, l2, n3, n1)
        end do
        n_faces = size(l1)
        allocate(faces(n_faces, 2))
        faces(:, 1) = l1
        faces(:, 2) = l2

        fi = 1
        fj = n_inflow_faces
        inflow_faces = [(i, i = fi, fj)]

        fi = fi + n_inflow_faces
        fj = fj + n_outflow_faces
        outflow_faces = [(i, i = fi, fj)]

        fi = fi + n_outflow_faces
        fj = fj + n_solid_faces
        solid_faces = [(i, i = fi, fj)]

        n_boundary_faces = n_inflow_faces + n_outflow_faces + n_solid_faces
        n_interior_faces = n_faces - n_boundary_faces

        fi = 1
        fj = n_boundary_faces
        boundary_faces = [(i, i = fi, fj)]

        fi = n_boundary_faces + 1
        fj = n_faces
        interior_faces = [(i, i = fi, fj)]
        
        ! cell_faces

        allocate(cell_faces(n_cells, 3))
        do i = 1, n_cells
            n1 = cells(i, 1)
            n2 = cells(i, 2)
            n3 = cells(i, 3)

            do j = 1, n_faces
                if (check_face(l1(j), l2(j), n1, n2)) then
                    cell_faces(i, 1) = j
                end if
                if (check_face(l1(j), l2(j), n2, n3)) then
                    cell_faces(i, 2) = j
                end if
                if (check_face(l1(j), l2(j), n3, n1)) then
                    cell_faces(i, 3) = j
                end if
            end do
        end do

        ! face_cells

        allocate(face_cells(n_faces, 2))
        do i = 1, n_faces
            l3 = [integer ::]
            do j = 1, n_cells
                if (any(cell_faces(j, :) == i)) then
                    l3 = [l3, j]
                end if
            end do        
            if (size(l3) == 1) then
                face_cells(i, 1) = l3(1)
                face_cells(i, 2) = l3(1)
            else
                face_cells(i, 1) = l3(1)
                face_cells(i, 2) = l3(2)
            end if
        end do

        ! Sf, nf, cf

        allocate(Sf(n_faces))
        allocate(nf(n_faces, 2))
        allocate(cf(n_faces, 2))
        allocate(face_faces(n_faces, 2))
        do i = 1, n_faces
            n1 = faces(i, 1)
            n2 = faces(i, 2)
            n1x = nodes(n1, 1)
            n1y = nodes(n1, 2)
            n2x = nodes(n2, 1)
            n2y = nodes(n2, 2)

            Sf(i) = sqrt((n2x - n1x)**2 + (n2y - n1y)**2)
            nf(i, :) = [(n2y - n1y)/Sf(i), -(n2x - n1x)/Sf(i)]
            cf(i, :) = [.5*(n2x + n1x), .5*(n2y + n1y)]

            c1 = face_cells(i, 1)
            c2 = face_cells(i, 2)

            do j = 1, 3
                if (cell_faces(c1, j) == i) then
                    face_faces(i, 1) = j
                end if
                if (cell_faces(c2, j) == i) then
                    face_faces(i, 2) = j
                end if
            end do
        end do

        ! Vc, cc, sc

        allocate(Vc(n_cells))
        allocate(cc(n_cells, 2))
        allocate(sc(n_cells, 3))
        allocate(cell_cells(n_cells, 3))
        do i = 1, n_cells
            n1 = cells(i, 1)
            n2 = cells(i, 2)
            n3 = cells(i, 3)
            n1x = nodes(n1, 1)
            n1y = nodes(n1, 2)
            n2x = nodes(n2, 1)
            n2y = nodes(n2, 2)
            n3x = nodes(n3, 1)
            n3y = nodes(n3, 2)

            Vc(i) = .5*abs(n1x*(n2y - n3y) + n2x*(n3y - n1y) + n3x*(n1y - n2y))
            cc(i, :) = [(n1x + n2x + n3x)/3, (n1y + n2y + n3y)/3]

            do j = 1, 3
                fj = cell_faces(i, j)

                if (i == face_cells(fj, 1)) then
                    sc(i, j) = 1.
                    cell_cells(i, j) = face_cells(fj, 2)
                else
                    sc(i, j) = -1.
                    cell_cells(i, j) = face_cells(fj, 1)
                end if
            end do
        end do

        ! rfc, rnc

        allocate(rfc(n_cells, 3))
        allocate(rnc(n_cells, 3))
        allocate(cell_nodes(n_cells, 3))
        allocate(cell_face_faces(n_cells, 3))
        do i = 1, n_cells
            n1 = cells(i, 1)
            n2 = cells(i, 2)
            n3 = cells(i, 3)

            do j = 1, 3
                fj = cell_faces(i, j)
                cj = cell_cells(i, j)

                na = faces(fj, 1)
                nb = faces(fj, 2)

                if (n1 /= na .and. n1 /= nb) then
                    nc = n1
                    cell_nodes(i, j) = nc
                end if
                if (n2 /= na .and. n2 /= nb) then
                    nc = n2
                    cell_nodes(i, j) = nc
                end if
                if (n3 /= na .and. n3 /= nb) then
                    nc = n3
                    cell_nodes(i, j) = nc
                end if

                cfx = cf(fj, 1)
                cfy = cf(fj, 2)
                ccx = cc(i, 1)
                ccy = cc(i, 2)
                ncx = nodes(nc, 1)
                ncy = nodes(nc, 2)
                
                rfc(i, j) = sqrt((cfx - ccx)**2 + (cfy - ccy)**2)
                rnc(i, j) = sqrt((ccx - ncx)**2 + (ccy - ncy)**2)

                cell_face_faces(i, j) = findloc(cell_faces(cj, :), fj, 1)
            end do
        end do

        ! wn

        allocate(wn(n_nodes, 10))
        allocate(dn(n_nodes, 10))
        allocate(node_cells(n_nodes, 10))
        allocate(n_node_cells(n_nodes))
        do i = 1, n_nodes
            k = 1
            do j = 1, n_cells
                if (any(cells(j, :) == i)) then
                    dn(i, k) = 1/norm2(cc(j, :) - nodes(i, :))  ! reciprocal
                    node_cells(i, k) = j
                    k = k + 1
                end if
            end do
            n_node_cells(i) = k - 1

            do j = 1, n_node_cells(i)
                wn(i, j) = dn(i, j)/sum(dn(i, :))
            end do
        end do

    end subroutine set_mesh

    subroutine set_face(la, lb, na, nb)

        integer, dimension(:), allocatable, intent(in out) :: la, lb
        integer, intent(in) :: na, nb
        
        integer, dimension(:), allocatable :: lna, lnb
        logical :: in
        
        allocate(lna, mold=la)
        allocate(lnb, mold=lb)
        lna(:) = na
        lnb(:) = nb

        in = any((la == lna) .and. (lb == lnb)) .or. any((la == lnb) .and. (lb == lna))
        if (.not. in) then
            la = [la, na]
            lb = [lb, nb]
        end if

    end subroutine set_face

    pure function check_face(la, lb, na, nb)

        integer, intent(in) :: la, lb, na, nb
        logical :: check_face

        if ((na == la .and. nb == lb) .or. (na == lb .and. nb == la)) then
            check_face = .true.
        else
            check_face = .false.
        end if
    
    end function check_face

    subroutine set_initial()
        
        integer :: i

        b(:) = 0.
        h(:) = .5
        qx(:) = 0.
        qy(:) = 0.

        do i = 1, n_cells
            if (sqrt((cc(i, 1) - 25.)**2 + (cc(i, 2) - 25.)**2) <= 10.) then
                h(i) = 2.
            end if
        end do
    
    end subroutine set_initial

    subroutine set_reconstruction()

        integer :: i, j, cj, fj, fcj, na, nb, nc
        real(precision) :: rfj, rnj, hmin, bLj, bRj, bmax, uL, vL
        real(precision) :: dhf, dhn, hLj, dzf, dzn, zLj, dqxf, dqxn, qxLj, dqyf, dqyn, qyLj
        
        ! gradient reconstruction

        do i = 1, n_nodes
            hn(i) = 0.
            qxn(i) = 0.
            qyn(i) = 0.
            zn(i) = 0.
            do j = 1, n_node_cells(i)
                cj = node_cells(i, j)

                hn(i) = hn(i) + wn(i, j)*h(cj)
                qxn(i) = qxn(i) + wn(i, j)*qx(cj)
                qyn(i) = qyn(i) + wn(i, j)*qy(cj)
                zn(i) = zn(i) + wn(i, j)*z(cj)
            end do
        end do

        do i = 1, n_cells
            do j = 1, 3
                fj = cell_faces(i, j)
                
                na = faces(fj, 1)
                nb = faces(fj, 2)
                nc = cell_nodes(i, j)
                
                rfj = rfc(i, j)
                rnj = rnc(i, j)

                dhf = (.5*(hn(na) + hn(nb)) - h(i))/rfj
                dhn = (h(i) - hn(nc))/rnj
                hLj = h(i) + rfj*get_limiter(dhf, dhn)

                dzf = (.5*(zn(na) + zn(nb)) - z(i))/rfj
                dzn = (z(i) - zn(nc))/rnj
                zLj = z(i) + rfj*get_limiter(dzf, dzn)

                bLj = zLj - hLj

                ! adaptive reconstruction

                hmin = min(abs(bLj - b(i)), .25*h(i))

                if (hLj <= hmin .or. h(i) <= dry) then
                    hL(i, j) = h(i)
                    qxL(i, j) = qx(i)
                    qyL(i, j) = qy(i)
                    zL(i, j) = z(i)
                    bL(i, j) = z(i) - h(i)
                else
                    dqxf = (.5*(qxn(na) + qxn(nb)) - qx(i))/rfj
                    dqxn = (qx(i) - qxn(nc))/rnj
                    qxLj = qx(i) + rfj*get_limiter(dqxf, dqxn)

                    dqyf = (.5*(qyn(na) + qyn(nb)) - qy(i))/rfj
                    dqyn = (qy(i) - qyn(nc))/rnj
                    qyLj = qy(i) + rfj*get_limiter(dqyf, dqyn)

                    hL(i, j) = hLj
                    qxL(i, j) = qxLj
                    qyL(i, j) = qyLj
                    zL(i, j) = zLj
                    bL(i, j) = bLj
                end if
            end do 
        end do

        ! hydrostatic reconstruction

        do i = 1, n_cells  ! separate loop
            do j = 1, 3
                cj = cell_cells(i, j)
                fcj = cell_face_faces(i, j)

                bLj = bL(i, j)
                bRj = bL(cj, fcj)  ! requires separate loop
                bmax = max(bLj, bRj)

                zLj = zL(i, j)
                uL = qxL(i, j)/hL(i, j)
                vL = qyL(i, j)/hL(i, j)

                hLj = max(0., zLj - bmax)
                qxLj = hLj*uL
                qyLj = hLj*vL

                hL(i, j) = hLj 
                qxL(i, j) = qxLj 
                qyL(i, j) = qyLj
            end do
        end do

    end subroutine set_reconstruction

    pure function get_limiter(da, db)

        real(precision), intent(in) :: da, db
        real(precision) :: get_limiter

        if (da*db > 0.) then
            get_limiter = ((da**2 + 1e-16)*db + (db**2 + 1e-16)*da)/(da**2 + db**2 + 2e-16)
        else
            get_limiter = 0.
        end if

    end function get_limiter

    subroutine set_convection()

        integer :: i, ci, cj, fi, fj
        real(precision) :: nfx, nfy, hLj, qxLj, qyLj, hRj, qxRj, qyRj
        real(precision) :: uL, vL, uhL, vhL, uR, vR, uhR, vhR, hs, uhs, sL, sR, ss
        real(precision), dimension(2) :: QhL, QhR, FhL, FhR, Fs

        do i = 1, n_faces
            ci = face_cells(i, 1)
            fi = face_faces(i, 1)
            nfx = nf(i, 1)  ! unsigned
            nfy = nf(i, 2)  ! unsigned

            hLj = hL(ci, fi)
            qxLj = qxL(ci, fi)
            qyLj = qyL(ci, fi)

            if (hLj == 0.) then  ! qxLj == 0. also
                uL = 0.
                vL = 0.
                uhL = 0.
                vhL = 0.
            else
                uL = qxLj/hLj
                vL = qyLj/hLj
                uhL = uL*nfx + vL*nfy
                vhL = -uL*nfy + vL*nfx
            end if

            if (any(i == solid_faces)) then
                F(i, 1) = 0.
                F(i, 2) = g*hLj**2*nfx/2.
                F(i, 3) = g*hLj**2*nfy/2.
            else if (any(i == inflow_faces)) then
                print *, 'inflow'
            else  ! outflow/interior faces
                cj = face_cells(i, 2)
                fj = face_faces(i, 2)

                ! wave speed estimates

                hRj = hL(cj, fj)
                qxRj = qxL(cj, fj)
                qyRj = qyL(cj, fj)

                if (hRj == 0.) then  ! qxRj == 0. also
                    uR = 0.
                    vR = 0.
                    uhR = 0.
                    vhR = 0.
                else
                    uR = qxRj/hRj
                    vR = qyRj/hRj
                    uhR = uR*nfx + vR*nfy
                    vhR = -uR*nfy + vR*nfx
                end if

                hs = (1./g)*(.5*(sqrt(g*hLj) + sqrt(g*hRj)) + .25*(uhL - uhR))**2
                uhs = .5*(uhL + uhR) + sqrt(g*hLj) - sqrt(g*hRj)

                if (hLj == 0.) then
                    sL = uhR - 2.*sqrt(g*hRj)
                else if (hLj > 0.) then
                    sL = min(uhL - sqrt(g*hLj), uhs - sqrt(g*hs))
                else
                    print *, 'negative depth error'
                end if

                if (hRj == 0.) then
                    sR = uhL + 2*sqrt(g*hLj)
                else if (hRj > 0.) then
                    sR = max(uhR + sqrt(g*hRj), uhs + sqrt(g*hs))
                else
                    print *, 'negative depth error'
                end if

                ss = (sL*hRj*(uhR - sR) - sR*hLj*(uhL - sL))/(hRj*(uhR - sR) - hLj*(uhL - sL))

                ! flux estimates

                if (sL >= 0.) then
                    F(i, 1) = qxLj*nfx + qyLj*nfy
                    F(i, 2) = (uL*qxLj + .5*g*hLj**2)*nfx + vL*qxLj*nfy
                    F(i, 3) = uL*qyLj*nfx + (vL*qyLj + .5*g*hLj**2)*nfy
                else if (sL < 0. .and. ss >= 0.) then
                    QhL = [hLj, qxLj*nfx + qyLj*nfy]
                    QhR = [hRj, qxRj*nfx + qyRj*nfy]

                    FhL = [hLj*uhL, uhL*(qxLj*nfx + qyLj*nfy) + .5*g*hLj**2]
                    FhR = [hRj*uhR, uhR*(qxRj*nfx + qyRj*nfy) + .5*g*hRj**2]
                    Fs = (sR*FhL - sL*FhR + sL*sR*(QhR - QhL))/(sR - sL)

                    F(i, 1) = Fs(1)
                    F(i, 2) = Fs(2)*nfx - vhL*Fs(1)*nfy
                    F(i, 3) = Fs(2)*nfy + vhL*Fs(1)*nfx
                else if (ss < 0. .and. sR >= 0.) then
                    QhL = [hLj, qxLj*nfx + qyLj*nfy]
                    QhR = [hRj, qxRj*nfx + qyRj*nfy]

                    FhL = [hLj*uhL, uhL*(qxLj*nfx + qyLj*nfy) + .5*g*hLj**2]
                    FhR = [hRj*uhR, uhR*(qxRj*nfx + qyRj*nfy) + .5*g*hRj**2]
                    Fs = (sR*FhL - sL*FhR + sL*sR*(QhR - QhL))/(sR - sL)

                    F(i, 1) = Fs(1)
                    F(i, 2) = Fs(2)*nfx - vhR*Fs(1)*nfy
                    F(i, 3) = Fs(2)*nfy + vhR*Fs(1)*nfx
                else if (sR < 0.) then
                    F(i, 1) = qxRj*nfx + qyRj*nfy
                    F(i, 2) = (uR*qxRj + .5*g*hRj**2)*nfx + vR*qxRj*nfy
                    F(i, 3) = uR*qyRj*nfx + (vR*qyRj + .5*g*hRj**2)*nfy
                else
                    print *, 'HLLC flux error'
                end if
            end if
        end do

    end subroutine set_convection

    subroutine set_slope()

        integer :: i, j, cj, fj, fcj
        real(precision) :: Vci, Sfj, nfx, nfy, scj, hLj, zLj, bLj, bRj, bmax, bmin

        Ks(:, :) = 0.
        do i = 1, n_cells
            Vci = Vc(i)

            do j = 1, 3
                cj = cell_cells(i, j)
                fj = cell_faces(i, j)
                fcj = cell_face_faces(i, j)
                scj = sc(i, j)
                nfx = nf(fj, 1)  ! unsigned
                nfy = nf(fj, 2)  ! unsigned
                Sfj = Sf(fj)

                hLj = hL(i, j)
                zLj = zL(i, j)
                bLj = bL(i, j)
                bRj = bL(cj, fcj)
                bmax = max(bLj, bRj)
                bmin = min(bmax, zLj)

                Ks(i, 1) = Ks(i, 1) - dt/Vci*Sfj*scj*F(fj, 1)
                Ks(i, 2) = Ks(i, 2) - dt/Vci*Sfj*scj*(F(fj, 2) + nfx*g*(hLj + h(i))*(bmin - b(i))/2.)
                Ks(i, 3) = Ks(i, 3) - dt/Vci*Sfj*scj*(F(fj, 3) + nfy*g*(hLj + h(i))*(bmin - b(i))/2.)
            end do

            h(i) = h(i) + Ks(i, 1)
            qx(i) = qx(i) + Ks(i, 2)
            qy(i) = qy(i) + Ks(i, 3)
        end do

    end subroutine set_slope

    subroutine set_friction()

        integer :: i
        real(precision) :: hi, qxi, qyi, ui, vi, Cfr, qh, Sfx, Sfy, Sbfx, Sbfy

        do i = 1, n_cells
            hi = h(i)
            qxi = qx(i)
            qyi = qy(i)
            ui = qxi/hi
            vi = qyi/hi

            Cfr = g*n**2/hi**(1./3.)
            qh = sqrt(qxi**2 + qyi**2)
            Sfx = -Cfr*ui*sqrt(ui**2 + vi**2)
            Sfy = -Cfr*vi*sqrt(ui**2 + vi**2)

            if (qh == 0.) then  ! qxi == 0. and qyi == 0. also
                Sbfx = 0.
                Sbfy = 0.
            else
                Sbfx = Sfx/(1. + dt*Cfr/hi**2*(qh + qxi**2/qh))
                Sbfy = Sfy/(1. + dt*Cfr/hi**2*(qh + qyi**2/qh))
            end if

            if (qxi >= 0.) then
                Sbfx = max(-qxi/dt, Sbfx)
            else
                Sbfx = min(-qxi/dt, Sbfx)
            end if

            if (qyi >= 0.) then
                Sbfy = max(-qyi/dt, Sbfy)
            else
                Sbfy = min(-qyi/dt, Sbfy)
            end if

            h(i) = h(i) + 0.
            qx(i) = qx(i) + dt*Sbfx
            qy(i) = qy(i) + dt*Sbfy
        end do

    end subroutine set_friction

    subroutine set_data()

        character(100) :: file
        integer :: i

        if (mod(it, interval) == 0) then
            write(file, '(a, i10.10, a)') 'z', it, '.txt'
            open(100, file=trim(data)//trim(file))
            do i = 1, n_cells
                write(100, *) cc(i, 1), cc(i, 2), b(i), z(i)
            end do
        end if

    end subroutine set_data

end program main