
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

# Create the final executable
#add_executable(sycl.out $<TARGET_OBJECTS:axpy.dp> $<TARGET_OBJECTS:main> $<TARGET_OBJECTS:read_data.cpp.dp> $<TARGET_OBJECTS:data_struct> $<TARGET_OBJECTS:VDW_Coulomb.dp>)
#add_executable(sycl.out  ${OBJECT_NAME})

# Link the object files to the executable
#target_link_libraries(sycl.out PRIVATE $<TARGET_OBJECTS:axpy.dp> $<TARGET_OBJECTS:read_data.cpp.dp> $<TARGET_OBJECTS:data_struct> $<TARGET_OBJECTS:VDW_Coulomb.dp>)
#target_link_libraries(sycl.out  ${OBJECT_NAME})

