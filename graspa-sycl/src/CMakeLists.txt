
# Specify the source files
set(SOURCES
    axpy.cpp
    main.cpp
    read_data.cpp
    data_struct.cpp
    VDW_Coulomb.cpp
)

# Create object files
foreach(SOURCE ${SOURCES})
    get_filename_component(OBJECT_NAME ${SOURCE} NAME_WE)
    add_library(${OBJECT_NAME} OBJECT ${SOURCE})
endforeach()

add_executable(sycl.out ${SOURCES})


set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
target_link_libraries(sycl.out)

