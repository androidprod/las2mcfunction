// main.cpp
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <chrono>
#include <ctime>
#include <mutex>
#include <cstdlib>
#include "block_palette.hpp"
#include "logger.hpp"
#if __has_include("miniz.h")
#include "miniz.h"
#define HAVE_MINIZ 1
#endif
namespace fs = std::filesystem;
using Point3 = std::array<double,3>;
static std::mutex g_mtx;
static std::string stripq(const std::string& s){ if(s.size()>=2 && ((s.front()=='\"'&&s.back()=='\"')||(s.front()=='\''&&s.back()=='\''))) return s.substr(1,s.size()-2); return s; }
static std::string tolower_s(std::string s){ std::transform(s.begin(),s.end(),s.begin(),[](unsigned char c){ return (char)std::tolower(c); }); return s; }
static fs::path mktempdir(const std::string& pref="las2mc_tmp"){ fs::path base = fs::temp_directory_path(); for(int i=0;i<1000;++i){ fs::path d = base/(pref + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "_" + std::to_string(i)); std::error_code ec; if(fs::create_directories(d,ec)) return d; } throw std::runtime_error("tempdir"); }
static void rmdirs(const fs::path& p){ std::error_code ec; fs::remove_all(p,ec); if(ec) logf(3,"Failed to remove temporary directory '%s': %s", p.string().c_str(), ec.message().c_str()); }
static Point3 transform_XZ_negY(const Point3& p){ return {p[0], p[2], -p[1]}; }
static Point3 apply_scale(const Point3& p,double s){ return {p[0]*s, p[1]*s, p[2]*s}; }
static bool load_xyz(const fs::path& p, std::vector<Point3>& out){ std::ifstream ifs(p.string()); if(!ifs) return false; std::string L; while(std::getline(ifs,L)){ if(L.empty()||L[0]=='#') continue; std::istringstream ss(L); double x,y,z; if(ss>>x>>y>>z) out.push_back({x,y,z}); } return !out.empty(); }
// ---- LAS helpers (basic LAS 1.2 header parse)
static inline uint16_t le16(const uint8_t* p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
static inline uint32_t le32(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static inline double le64d(const uint8_t* p) { double v; std::memcpy(&v, p, 8); return v; }

struct LasHeader { uint16_t header_size; uint32_t offset_to_point_data; uint8_t point_format; uint16_t rec_len; uint32_t num_points; double xs, ys, zs, xo, yo, zo; };

static bool read_las_header(const std::string& path, LasHeader& H) {
    std::ifstream ifs(path, std::ios::binary); if (!ifs) return false;
    std::vector<uint8_t> buf(227); ifs.read((char*)buf.data(), buf.size()); if ((size_t)ifs.gcount() < buf.size()) return false;
    if (!(buf[0] == 'L' && buf[1] == 'A' && buf[2] == 'S' && buf[3] == 'F')) return false;
    H.header_size = le16(&buf[94]); H.offset_to_point_data = le32(&buf[96]); H.point_format = buf[104]; H.rec_len = le16(&buf[105]);
    H.num_points = le32(&buf[107]); H.xs = le64d(&buf[131]); H.ys = le64d(&buf[139]); H.zs = le64d(&buf[147]);
    H.xo = le64d(&buf[155]); H.yo = le64d(&buf[163]); H.zo = le64d(&buf[171]); return true;
}

// Load LAS: outputs coordinates (X,Y,Z) into coords and corresponding RGB (0..255) into colors.
static bool load_points_from_las(const fs::path& p, std::vector<Point3>& coords, std::vector<Point3>& colors) {
    LasHeader H; if (!read_las_header(p.string(), H)) return false;
    std::ifstream ifs(p.string(), std::ios::binary); if (!ifs) return false;
    ifs.seekg(H.offset_to_point_data, std::ios::beg);
    std::vector<uint8_t> rec(H.rec_len);
    std::unordered_set<uint64_t> seen; seen.reserve(100000);
    coords.clear(); colors.clear(); coords.reserve(std::min<size_t>(H.num_points, 1000000));
    for (uint32_t i = 0; i < H.num_points; ++i) {
        ifs.read((char*)rec.data(), rec.size()); if ((size_t)ifs.gcount() < rec.size()) break;
        int32_t xi = (int32_t)le32(&rec[0]); int32_t yi = (int32_t)le32(&rec[4]); int32_t zi = (int32_t)le32(&rec[8]);
        int X = (int)std::llround((xi * H.xs + H.xo));
        int Y = (int)std::llround((yi * H.ys + H.yo));
        int Z = (int)std::llround((zi * H.zs + H.zo));
        uint64_t key = (((uint64_t)(uint32_t)X) << 42) ^ (((uint64_t)(uint32_t)Y) << 21) ^ (uint32_t)Z;
        if (seen.find(key) != seen.end()) continue;
        seen.insert(key);
        int r = 255, g = 255, b = 255;
        if (H.point_format == 2 || H.point_format == 3) {
            size_t base = H.rec_len >= 6 ? H.rec_len - 6 : 0;
            uint16_t rr = le16(&rec[base]); uint16_t gg = le16(&rec[base + 2]); uint16_t bb = le16(&rec[base + 4]);
            r = (rr >> 8); g = (gg >> 8); b = (bb >> 8);
            r = std::clamp(r, 0, 255); g = std::clamp(g, 0, 255); b = std::clamp(b, 0, 255);
        }
        coords.push_back({(double)X, (double)Y, (double)Z});
        colors.push_back({(double)r, (double)g, (double)b});
    }
    return !coords.empty();
}
static bool run_cmd(const std::string& cmd){ int r = std::system(cmd.c_str()); return r==0; }
static bool convert_laz_to_las_external(const fs::path& laz,const fs::path& las){
#ifdef _WIN32
    std::string cmd = "laszip -i \"" + laz.string() + "\" -o \"" + las.string() + "\"";
#else
    std::string cmd = "laszip -i '" + laz.string() + "' -o '" + las.string() + "'";
#endif
    logf(0,"exec: %s",cmd.c_str());
    return run_cmd(cmd) && fs::exists(las);
}
#ifdef HAVE_MINIZ
static bool extract_zip(const fs::path& zipfile,const fs::path& outdir){
    mz_zip_archive zip; memset(&zip,0,sizeof(zip));
    if(!mz_zip_reader_init_file(&zip, zipfile.string().c_str(), 0)){
        logf(3,"mz init failed: %s",zipfile.string().c_str());
        return false;
    }
    fs::create_directories(outdir);
    mz_uint n = mz_zip_reader_get_num_files(&zip);
    for(mz_uint i=0;i<n;i++){
        mz_zip_archive_file_stat st;
        if(!mz_zip_reader_file_stat(&zip,i,&st)) continue;
        if(mz_zip_reader_is_file_a_directory(&zip,i)) continue;
        fs::path out = outdir / fs::path(st.m_filename).filename();
        fs::create_directories(out.parent_path());
        if(!mz_zip_reader_extract_to_file(&zip,i,out.string().c_str(),0))
            logf(2,"zip extract fail: %s",st.m_filename);
        else
            logf(0,"extracted: %s",out.string().c_str());
    }
    mz_zip_reader_end(&zip);
    return true;
}
#else
static bool extract_zip(const fs::path&,const fs::path&){ logf(3,"ZIP not supported (miniz missing)"); return false; }
#endif
static bool write_mc(const fs::path& out,const std::vector<Point3>& pts,const std::vector<int>& idx)
{
    std::ofstream f(out.string());
    if(!f){ logf(ERR,"write fail %s",out.string().c_str()); return false; }
    f << "# generated\n# points:" << pts.size() << "\n";
    for(size_t i=0;i<pts.size();++i){
        auto &p = pts[i];
        long X = (long)std::lround(p[0]);
        long Y = (long)std::lround(p[1]);
        long Z = (long)std::lround(p[2]);
        const char* bn = "minecraft:stone";
        if(i < idx.size()) bn = block_palette::block_name(idx[i]);
        f << "setblock " << X << " " << Y << " " << Z << " " << bn << "\n";
    }
    return true;
}

// Write .mcfunction files split into fixed-size chunks (10000 lines per file)
static bool write_mc_chunked(const fs::path& basepath, const std::string& base, const std::vector<Point3>& pts, const std::vector<int>& idx)
{
    constexpr int MAX_LINES = 10000;
    std::error_code ec;
    fs::create_directories(basepath, ec);
    if (ec) { logf(3, "Failed to create output dir: %s", basepath.string().c_str()); return false; }

    int chunk = 1;
    int line = 0;
    std::string outname;
    std::ofstream ofs;
    auto open_chunk = [&](int c)->bool {
        if (ofs.is_open()) ofs.close();
        fs::path p = basepath / (base + "_slice_" + std::to_string(c) + ".mcfunction");
        ofs.open(p.string(), std::ios::out);
        return ofs.is_open();
    };

    if (!open_chunk(chunk)) return false;

    // Buffer lines to reduce syscall overhead
    std::string buffer;
    buffer.reserve(1024 * 1024);

    for (size_t i = 0; i < pts.size(); ++i) {
        auto &p = pts[i];
        long X = (long)std::lround(p[0]), Y = (long)std::lround(p[1]), Z = (long)std::lround(p[2]);
        const char* bn = "minecraft:stone";
        if (i < idx.size()) bn = block_palette::block_name(idx[i]);
        buffer.append("setblock "); buffer.append(std::to_string(X)); buffer.push_back(' ');
        buffer.append(std::to_string(Y)); buffer.push_back(' ');
        buffer.append(std::to_string(Z)); buffer.append(" "); buffer.append(bn); buffer.append("\n");
        ++line;
        if (line >= MAX_LINES) {
            // flush and open next
            ofs << buffer;
            ofs.flush();
            buffer.clear();
            ++chunk; line = 0;
            if (!open_chunk(chunk)) return false;
        }
    }
    if (!buffer.empty() && ofs.is_open()) ofs << buffer;
    if (ofs.is_open()) ofs.close();
    return true;
}

static bool process_single(const fs::path& in,const fs::path& out,double scale,bool prefer_cuda)
{
    logf(INF,"processing %s -> %s (scale=%.3f,cuda=%s)", in.string().c_str(), out.string().c_str(), scale, prefer_cuda?"yes":"no");
    std::vector<Point3> pts;
    bool ok = false;
    auto ext = tolower_s(in.extension().string());
    std::vector<Point3> colors;
    if(ext==".las"){
        ok = load_points_from_las(in, pts, colors);
        if(!ok){ logf(ERR,"Error: Failed to read LAS file %s.", in.string().c_str()); return false; }
    }
    else if(ext==".xyz"||ext==".txt"){
        ok = load_xyz(in, pts);
        if(!ok){ logf(ERR,"Error: Failed to load XYZ/text points from %s.", in.string().c_str()); return false; }
    }
    else {
        logf(ERR,"Error: Unsupported file extension for direct processing: %s. Expected .las, .xyz, or .txt for individual files.", in.string().c_str());
        return false;
    }
    // If colors were provided (from LAS), build color points, otherwise use coordinates as color proxy
    for(size_t i=0;i<pts.size();++i){
        pts[i] = transform_XZ_negY(pts[i]);
        if(scale!=1.0) pts[i] = apply_scale(pts[i],scale);
    }
    std::vector<int> idx;
    if(!colors.empty()){
        std::vector<Point3> color_pts; color_pts.reserve(colors.size());
        for(size_t i=0;i<colors.size();++i) color_pts.push_back(colors[i]);
        block_palette::map_blocks(color_pts, idx, prefer_cuda);
    } else {
        block_palette::map_blocks(pts, idx, prefer_cuda);
    }
    fs::path outp = out;
    if (fs::is_directory(outp) || outp.string().back() == '\\' || outp.string().back() == '/') {
        fs::create_directories(outp);
        // write chunked files into the directory using the input stem as base name
        std::string base = in.stem().string();
        return write_mc_chunked(outp, base, pts, idx);
    }
    // If outp is a file path, ensure parent dir exists and write single file (but still chunk if large)
    fs::path parent = outp.parent_path(); if (!parent.empty()) fs::create_directories(parent);
    std::string base = outp.stem().string();
    // write into parent using provided filename base
    return write_mc_chunked(parent, base, pts, idx);
}

static bool process_path(const fs::path& inp,const fs::path& out,double scale,bool prefer_cuda)
{
    if(!fs::exists(inp)){ logf(ERR,"not found %s",inp.string().c_str()); return false; }
    if(fs::is_directory(inp)){
        bool ok=true;
        for(auto &e:fs::recursive_directory_iterator(inp)) if(e.is_regular_file()) ok = process_path(e.path(), out, scale, prefer_cuda) && ok;
        return ok;
    }
    auto ext = tolower_s(inp.extension().string());
    if(ext==".las") return process_single(inp,out,scale,prefer_cuda);
    if(ext==".laz"){
        fs::path tmpd = mktempdir();
        fs::path las = tmpd/(inp.stem().string()+".las");
        bool conv = convert_laz_to_las_external(inp,las);
        bool r=false;
        if(conv) r = process_single(las,out,scale,prefer_cuda);
        else logf(ERR,"LAZ->LAS failed %s",inp.string().c_str());
        rmdirs(tmpd);
        return r;
    }
    if(ext==".zip"){
        fs::path tmpd = mktempdir();
        bool ex = extract_zip(inp,tmpd);
        bool ok = false;
        if(ex){
            ok = true;
            for(auto &e:fs::recursive_directory_iterator(tmpd)){
                if(!e.is_regular_file()) continue;
                std::string ee = tolower_s(e.path().extension().string());
                if(ee==".las"||ee==".xyz"||ee==".txt") ok = process_single(e.path(), out, scale, prefer_cuda) && ok;
                else if(ee==".laz"){
                    fs::path las = mktempdir()/(e.path().stem().string()+".las");
                    if(convert_laz_to_las_external(e.path(),las)) ok = process_single(las,out,scale,prefer_cuda)&&ok;
                    else { logf(WARN,"skip laz in zip: %s",e.path().string().c_str()); ok=false; }
                }
            }
        }
        rmdirs(tmpd);
        return ok;
    }
    logf(WARN,"unsupported %s",inp.string().c_str());
    return false;
}
static void interactive(){
    double scale=1.0; bool prefer_cuda=false;
    logf(INF,"Interactive mode. Type 'help'");
    std::string line;
    while (std::cout << "las2mcfunction> ", std::getline(std::cin, line)) {
        if(line.empty()) continue;
        std::istringstream ss(line); std::string cmd; ss>>cmd;
        std::string lcmd=tolower_s(cmd);
        if(lcmd=="exit"||lcmd=="quit") break;
        if(lcmd=="help"){ std::cout<<"run <in> <out>\nset scale <n>\nset cuda <on|off>\nstatus\nexit\n"; continue; }
        if(lcmd=="set"){ std::string k,v; ss>>k>>v; if(k=="scale"){ try{ scale=std::stod(v); logf(INF,"scale=%.3f",scale);}catch(...){logf(WARN,"bad scale");}} else if(k=="cuda"){ prefer_cuda=(tolower_s(v)=="on"); logf(INF,"cuda=%s",prefer_cuda?"on":"off"); } continue; }
        if(lcmd=="status"){ logf(INF,"scale=%.3f cuda=%s",scale, prefer_cuda?"on":"off"); continue; }
        if(lcmd=="run"){ std::string a,b; ss>>std::ws; std::getline(ss,a,' '); if(a.empty()){ ss>>a; } ss>>std::ws; std::getline(ss,b); if(b.empty()) { ss>>b; } a=stripq(a); b=stripq(b); if(a.empty()||b.empty()){ logf(WARN,"need in/out"); continue; } process_path(fs::path(a), fs::path(b), scale, prefer_cuda); continue; }
        logf(WARN,"unknown cmd");
    }
}
int main(int argc,char**argv){
    if(argc<=1){ interactive(); return 0; }
    std::string in=stripq(argv[1]); std::string out=(argc>=3?stripq(argv[2]):".");
    double scale=1.0; bool prefer_cuda=false;
    for(int i=3;i<argc;i++){ std::string s=argv[i]; if(s=="--scale"&&i+1<argc){ try{ scale=std::stod(argv[++i]); }catch(...){ } } else if(s=="--no-cuda") prefer_cuda=false; else if(s=="--cuda") prefer_cuda=true; }
    bool ok = process_path(fs::path(in), fs::path(out), scale, prefer_cuda);
    return ok?0:2;
}
