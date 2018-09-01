function G = edge_detect(im)
%     G = canny_edge(rgb2gray(im));
    [~,~,G]=edge_canny(rgb2gray(im),'canny');

end
