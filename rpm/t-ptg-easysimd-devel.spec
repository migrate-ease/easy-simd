##############################################################
Name: t-ptg-easysimd-devel
Version: 1.0.0
Release: %(echo $RELEASE)%{?dist}
Group: AnolisOS/application
License: Commercial
Summary: A high-performance cross-platform vectorization acceleration library 
AutoReqProv: none
Requires: meson
Requires: ninja-build
Prefix:         /usr/local
%define _prefix /usr/local

%description
A high-performance cross-platform vectorization acceleration library 

%global _build_id_links none

# prepare your files

%build
cd $OLDPWD/..
rm -rf build
CFLAGS="-march=armv8.5-a+crc+sve2+sha2+sha3+sve2-sha3+sve2-aes+sve2-bitperm -msve-vector-bits=128" CXXFLAGS="-march=armv8.5-a+crc+sve2+sha2+sha3+sve2-sha3+sve2-aes+sve2-bitperm -msve-vector-bits=128" meson build
ninja -C build

%install
cd $OLDPWD/..
mkdir -p $RPM_BUILD_ROOT/%{_prefix}
mkdir -p $RPM_BUILD_ROOT/%{_prefix}/include/easysimd/
cp -rf easysimd $RPM_BUILD_ROOT/%{_prefix}/include/easysimd/
cp -rf test $RPM_BUILD_ROOT/%{_prefix}/include/easysimd/
cp -rf example $RPM_BUILD_ROOT/%{_prefix}/include/easysimd/
cp -rf build.sh $RPM_BUILD_ROOT/%{_prefix}/include/easysimd/
cp -rf COPYING $RPM_BUILD_ROOT/%{_prefix}/include/easysimd/
cp -rf meson_options.txt $RPM_BUILD_ROOT/%{_prefix}/include/easysimd/
cp -rf meson.build $RPM_BUILD_ROOT/%{_prefix}/include/easysimd/
cp -rf README.md $RPM_BUILD_ROOT/%{_prefix}/include/easysimd/

# package infomation
%files
%defattr(-,root,root)
/usr/local/include/easysimd/*

%post
%postun

%changelog
* Thu Oct 13 2022 wudinggui 
- add spec of t-ptg-easysimd
