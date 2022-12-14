
#ifndef _CSR_HPP
#define _CSR_HPP

#include "types.h"
#include <iostream>
#include <cstring>
#include <vector>
using namespace std;
// ================ CLASS CSR ================
/**
* @brief CSR Format
* @par Sample:
* @code
*   //sample code here
* @endcode
* @note Limitations:
*/
template<typename T_data, class T_attr=bool> class CSR
{
private:
    T_data *_data;
    bool _with_attr;
    T_attr *_attr;
    T_CSR_count _item_count = 0, _item_capacity = 4; // for saving attributes
    T_CSR_cap *_offs;
    
    T_CSR_cap _size = 0;     // unit: counts not bytes, for saving raw data
    T_CSR_cap _capacity = 0; // unit: counts not bytes
    
    void _Realloc(T_CSR_cap new_capacity) {
        // new_capacity: capacity in counts
        _data = (T_data*)realloc(_data, new_capacity * sizeof(T_data));
    }
    
public:
    float size_incre_ratio = 1.5;
    
    CSR(bool with_attr = false, T_CSR_cap initial_capacity = 8) {
        _with_attr = with_attr;
        if (with_attr) _attr = new T_attr[_item_capacity]();//
        _offs = new T_CSR_cap[_item_capacity+1]();//
        _offs[0] = 0;
        _capacity = initial_capacity;
        _data = new T_data[_capacity]();//
    }
    CSR(const CSR<T_data, T_attr>& f) {
        _item_capacity = _item_count = f._item_count;
        if (f.has_attr()) {
            _attr = new T_attr[_item_capacity]();//
            _with_attr = true;
            memcpy(_attr, f._attr, sizeof(T_attr) * _item_count);
        }
        _offs = new T_CSR_cap[_item_capacity+1]();//?
        memcpy(_offs, f._offs, sizeof(T_CSR_cap)*(_item_count+1));
        _size = _capacity = f._size;
        _data = new T_data[_size]();//
        memcpy(_data, f._data, sizeof(T_data)*_size);
    }
    ~CSR(void) {
        delete [] _offs;//
        delete [] _data;//
        if (_with_attr) delete [] _attr;//
    }
    
    // append with no attr
    void append(const T_data *new_data, T_CSR_cap new_data_size) {
        // the unit of new_data_size is count not byte
        if (new_data_size + _size >= _capacity) {
            T_CSR_cap new_capacity = (T_CSR_cap)(double(_capacity)*size_incre_ratio);
            _capacity = new_data_size + _size > new_capacity ? new_data_size + _size + 1 : new_capacity;
            _Realloc(_capacity);
        }
        if (_item_count == _item_capacity-1) {
            _offs = (T_CSR_cap *)realloc(_offs, sizeof(T_CSR_cap) * (2*_item_capacity+1));
            if (_with_attr) _attr = (T_attr *)realloc(_attr, sizeof(T_attr) * (2*_item_capacity));
            _item_capacity = 2*_item_capacity;
        }
        memcpy(_data + _offs[_item_count], new_data, new_data_size * sizeof(T_data));
        _size += new_data_size;
        _offs[_item_count+1] = _size;
        _item_count += 1;
    }

    // append with attr
    void append(const T_data *new_data, T_CSR_cap new_data_size, T_attr attr) {
        // the unit of new_data_size is count not byte
        if (new_data_size + _size >= _capacity) {
            T_CSR_cap new_capacity = (T_CSR_cap)(double(_capacity)*size_incre_ratio);
            _capacity = new_data_size + _size > new_capacity ? new_data_size + _size + 1 : new_capacity;
            _Realloc(_capacity);
        }
        if (_item_count == _item_capacity-1) {
            _offs = (T_CSR_cap *)realloc(_offs, sizeof(T_CSR_cap) * (2*_item_capacity+1));
            if (_with_attr) _attr = (T_attr *)realloc(_attr, sizeof(T_attr) * (2*_item_capacity));
            _item_capacity = 2*_item_capacity;
        }
        memcpy(_data + _offs[_item_count], new_data, new_data_size * sizeof(T_data));
        _size += new_data_size;
        if (_with_attr) _attr[_item_count] = attr;
        _offs[_item_count+1] = _size;
        _item_count += 1;
    }
    void append(CSR<T_data, T_attr> new_data) {
        T_CSR_cap new_data_size = new_data.size();
        if (new_data_size + _size >= _capacity) {
            T_CSR_cap new_capacity = (T_CSR_cap)(double(_capacity)*size_incre_ratio);
            _capacity = new_data_size + _size > new_capacity ? new_data_size + _size + 1 : new_capacity;
            _Realloc(_capacity);
        }
        if (_item_count + new_data.items() >= _item_capacity) {
            // should multiply 2 below to avoid float size.
            int new_size = _item_count+new_data.items() > 2*_item_capacity ? _item_count+new_data.items() : 2*_item_capacity;
            _offs = (T_CSR_cap *)realloc(_offs, sizeof(T_CSR_cap) * (new_size+1));
            if (_with_attr) _attr = (T_attr *)realloc(_attr, sizeof(T_attr) * (new_size));
            _item_capacity = new_size;
        }
        memcpy(_data + _offs[_item_count], new_data.get_raw_data(), new_data_size * sizeof(T_data));
        memcpy(&_offs[_item_count+1], new_data.get_raw_offs()+1, new_data.items() * sizeof(T_CSR_cap));
        for(T_CSR_count i=_item_count+1; i<_item_count+new_data.items()+1; i++)
            _offs[i] += _offs[_item_count]; // add the offset value for new elements
        if (_with_attr && new_data.has_attr()) 
            memcpy(&_attr[_item_count], new_data.get_raw_attr(), new_data.items() * sizeof(T_attr));
        _size += new_data_size;
        _item_count += new_data.items();
    }
    
    T_CSR_cap capacity() { return _capacity; }
    T_CSR_cap size() { return _size; }
    T_CSR_count items() { return _item_count; }
    short dtype() { return sizeof(T_data); }
    
    void debug_info() {
        cout << "ITEMS\t" << this -> items() << "/" << _item_capacity << endl;
        for (int i=0; i<_item_count; i++) {
            cout << "offs="<< _offs[i+1];
            if (has_attr()) cout << " attr=" << _attr[i];
            cout << endl;
        }
        cout << "SIZE\t" << this -> size() << "/" << this -> capacity() << endl;
        cout << "DTYPE\t" <<this -> dtype() << endl;
        for (T_CSR_cap i=0; i<_size; i++) cout<<_data[i]<<" ";
        cout << endl;
    }
    T_data* get_raw_data() { return _data; }
    T_CSR_cap* get_raw_offs() { return _offs; }
    T_data* fetch_raw_data() {
        T_data* res = new T_data[_size];
        memcpy(res, _data, _size*sizeof(T_data));
        return res;
    }
    T_CSR_cap* fetch_raw_offs() {
        T_CSR_cap* res = new T_CSR_cap[_item_count+1];
        memcpy(res, _offs, (_item_count+1)*sizeof(T_CSR_cap));
        return res;
    }
    
    bool has_attr() { return _with_attr; }
    T_attr* get_raw_attr() { return _attr; }
    
    int get_item(int idx, _out_ T_data *data_buffer) {
        if (idx > this->size()) return -1;
        memcpy(data_buffer, &(this->_data[this->_offs[idx]]), sizeof(T_data) * (this->_offs[idx+1] - this->_offs[idx]));
        return 0;
    }
    T_attr get_attr(int idx) {
        if (idx > this->size()) {
            cerr << "[ERROR] CSR.get_attr(idx) got wrong idx " << idx << endl;
            exit(1);
        }
        return _attr[idx];
    }
};


/*

// ================ CLASS ReadLoader ================
class ReadLoader {
private:
    static char* _getline(FILE* fp, char *buffer, int buffer_size, _out_ string &line);
public:
    static const int LOAD_BUF_SIZE = 1048576 * 10;
    static void LoadReadsToVector(const char* filename, _out_ vector<string> &reads, int k = 1);
    static void LoadReadsToCSR(const char* filename, _out_ CSR<char> &reads, int k = 1);
    static void Vector2CSR(vector<string> &reads, CSR<char> &csr);
};


// ==================================================
// ================ CLASS ReadLoader ================
// ==================================================
char* ReadLoader::_getline(FILE* fp, char *buffer, int buffer_size, _out_ string &line) {
    line.clear();
    char* res_flag = fgets(buffer, buffer_size-1, fp);
    char* t = res_flag;
    while (buffer[strlen(buffer)-1] != '\r' && buffer[strlen(buffer)-1] != '\n' && t && !feof(fp)) {
        line += buffer;
        t = fgets(buffer, buffer_size-1, fp);
    } 
    for (int i = strlen(buffer)-1; i>=0; i--) {
        if (buffer[i] == '\r' || buffer[i] == '\n') buffer[i] = 0;
        else break;
    }
    line += buffer;
    return res_flag;
}

void ReadLoader::LoadReadsToVector(const char* filename, _out_ vector<string> &reads, int k) {//k=1
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        cerr << "Error when open " << filename << "." << endl;
        exit(1);
    }
    string line;
    char *read_buffer = new char[LOAD_BUF_SIZE];
    while(_getline(fp, read_buffer, LOAD_BUF_SIZE, line)) {
        if (line.length()>=k) reads.push_back(line);
    }
    fclose(fp);
    return;
}

void ReadLoader::LoadReadsToCSR(const char* filename, _out_ CSR<char> &reads, int k) {//k=1
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        cerr << "Error when open " << filename << "." << endl;
        exit(1);
    }
    string line;
    char *read_buffer = new char[LOAD_BUF_SIZE];
    while(_getline(fp, read_buffer, LOAD_BUF_SIZE, line)) {
        if (line.length()>=k) reads.append(line.c_str(), line.length());
    }
    fclose(fp);
    return;
}

void ReadLoader::Vector2CSR(vector<string> &reads, CSR<char> &csr) {
    for(string read: reads) {
        csr.append(read.c_str(), read.length());
    }
}

*/

#endif