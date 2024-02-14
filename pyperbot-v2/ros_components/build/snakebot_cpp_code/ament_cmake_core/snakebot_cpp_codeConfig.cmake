# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_snakebot_cpp_code_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED snakebot_cpp_code_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(snakebot_cpp_code_FOUND FALSE)
  elseif(NOT snakebot_cpp_code_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(snakebot_cpp_code_FOUND FALSE)
  endif()
  return()
endif()
set(_snakebot_cpp_code_CONFIG_INCLUDED TRUE)

# output package information
if(NOT snakebot_cpp_code_FIND_QUIETLY)
  message(STATUS "Found snakebot_cpp_code: 0.0.0 (${snakebot_cpp_code_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'snakebot_cpp_code' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT snakebot_cpp_code_DEPRECATED_QUIET)
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(snakebot_cpp_code_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${snakebot_cpp_code_DIR}/${_extra}")
endforeach()
