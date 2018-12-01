function [] = ProduceDir( )
%该函数用于批量产生文件夹%
for m=1:40
    mkdir('./image/',num2str(m)); 
end
end

   