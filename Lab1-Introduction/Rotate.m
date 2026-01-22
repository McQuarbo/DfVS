function Out = Rotated(In, Theta)
%ROTATE Rotate a grayscale image by Theta radians about its centre.
%   Uses reverse mapping + nearest neighbour. Output same size as input.
%   Pixels mapping from outside input are set to 0 (black).

    % Ensure numeric type for calculations
    In = double(In);

    [H, W] = size(In);

    % Centre of image (x = col, y = row)
    cx = (W + 1) / 2;
    cy = (H + 1) / 2;

    % Output initialised to black
    Out = zeros(H, W);

    % Forward rotation matrix A (as per the sheet)
    A = [ cos(Theta),  sin(Theta);
         -sin(Theta),  cos(Theta)];

    % Inverse for reverse mapping
    Ainv = inv(A);

    % Loop over destination pixels
    for y_d = 1:H
        for x_d = 1:W

            % Shift destination coord to centre
            xd = x_d - cx;
            yd = y_d - cy;

            % Reverse map back to source (still centred coords)
            src = Ainv * [xd; yd];

            % Shift back to image coords
            x_s = src(1) + cx;
            y_s = src(2) + cy;

            % Nearest neighbour
            x_nn = round(x_s);
            y_nn = round(y_s);

            % If inside bounds, copy pixel; else leave as 0
            if x_nn >= 1 && x_nn <= W && y_nn >= 1 && y_nn <= H
                Out(y_d, x_d) = In(y_nn, x_nn);
            end
        end
    end

    % (Optional) cast back to original type if you want:
    % Out = cast(Out, class(In));

end

load('clown.mat')      % gives variable "clown"
theta = pi/6;          % 30 degrees
rot = Rotated(clown, theta);

figure; imshow(clown, []); colormap gray; title('Original');
figure; imshow(rot,   []); colormap gray; title('Rotated');

