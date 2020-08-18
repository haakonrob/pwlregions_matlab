test_dir = fileparts(which(mfilename));

files = dir(test_dir);

for i = 1:length(files)
    f = files(i);
    [~, name, ext] = fileparts(f.name);
    
    if f.isdir || strcmp(ext, '.m~') || ismember(name, {'.','..',mfilename})
        continue
    end
    
    disp('============================');
    disp(f.name);
    disp('============================');
    try
        run(f.name)
        disp([f.name, "..... Pass"])  
    catch e
        disp([f.name, "..... Fail"])
        fprintf(1,'Error:\n%s\n\n',e.message);
    end
    disp('----------------------------');
end
